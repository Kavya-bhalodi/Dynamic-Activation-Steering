import os, json, gc, time, re, math
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import defaultdict
from steering_utils import BaseConfig, init_environment, load_model, load_steering_vectors, compute_gaussian_weights, compute_per_layer_alphas, generate_responses_batched
init_environment()
import os
import json
import time
import gc
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import torch
import h5py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class Config(BaseConfig):
    DATASET_PATH = 'datasets/pipeline_test/responses/checkpoint.json'
    OUTPUT_DIR = 'activations_output'
    MODEL_NAME = 'NousResearch/Meta-Llama-3-8B-Instruct'
    LAYERS_TO_EXTRACT = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 31]
    MAX_LENGTH = 1024
    CHECKPOINT_EVERY = 50
    EXTRACT_REGION = 'scratchpad'
    USE_4BIT = True
    HF_TOKEN = None

class ActivationExtractor:

    def __init__(self, model, layers_to_extract: List[int], device: str='cuda'):
        self.model = model
        self.layers_to_extract = layers_to_extract
        self.device = device
        self.activations: Dict[str, List] = defaultdict(list)
        self.hooks = []

    def _create_hook(self, layer_name: str):

        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            self.activations[layer_name].append(hidden_states.detach().cpu())
        return hook_fn

    def register_hooks(self):
        for layer_idx in self.layers_to_extract:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._create_hook(f'layer_{layer_idx}'))
            self.hooks.append(hook)
        print(f'  Registered hooks on {len(self.layers_to_extract)} layers: {self.layers_to_extract}')

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self):
        self.activations = defaultdict(list)

def load_checkpoint_dataset(path: str) -> List[Dict]:
    print(f'Loading dataset from {path}...')
    with open(path, 'r') as f:
        data = json.load(f)
    responses = data['responses']
    print(f'  Total entries: {len(responses)}')
    valid = [r for r in responses if r.get('honest_success') and r.get('scheming_success') and r.get('honest_response') and r.get('scheming_response')]
    print(f'  Valid pairs (both honest+scheming): {len(valid)}')
    from collections import Counter
    cats = Counter((r['category'] for r in valid))
    for cat, count in sorted(cats.items()):
        print(f'    {cat}: {count}')
    return valid

def find_scratchpad_token_range(token_ids: List[int], tokenizer, search_from: int=0) -> Tuple[Optional[int], Optional[int]]:
    tokens_text = [tokenizer.decode([tid]) for tid in token_ids]
    start_idx = None
    end_idx = None
    for i in range(search_from, len(tokens_text)):
        tok = tokens_text[i].lower()
        if 'scratchpad>' in tok and '/' not in tok and (start_idx is None):
            start_idx = i + 1
        elif '/scratchpad>' in tok and start_idx is not None:
            end_idx = i
            break
    return (start_idx, end_idx)

def find_response_start(full_text: str, prompt: str, tokenizer) -> int:
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return len(prompt_token_ids) + 1

def extract_region_activations(activations: Dict[str, List[torch.Tensor]], token_ids: List[int], tokenizer, full_text: str, prompt: str, response: str, extract_region: str='scratchpad') -> Dict[str, torch.Tensor]:
    result = {}
    response_start = find_response_start(full_text, prompt, tokenizer)
    if extract_region == 'scratchpad':
        start_idx, end_idx = find_scratchpad_token_range(token_ids, tokenizer, search_from=response_start)
        if start_idx is not None and end_idx is not None and (end_idx > start_idx):
            for layer_name, acts_list in activations.items():
                acts = acts_list[0]
                region_acts = acts[0, start_idx:end_idx, :]
                result[layer_name] = region_acts.mean(dim=0)
            return result
        else:
            extract_region = 'response'
    if extract_region == 'response':
        for layer_name, acts_list in activations.items():
            acts = acts_list[0]
            seq_len = acts.shape[1]
            actual_start = min(response_start, seq_len - 1)
            region_acts = acts[0, actual_start:, :]
            result[layer_name] = region_acts.mean(dim=0)
        return result
    for layer_name, acts_list in activations.items():
        acts = acts_list[0]
        result[layer_name] = acts[0].mean(dim=0)
    return result

def extract_single_completion(model, tokenizer, extractor: ActivationExtractor, prompt: str, response: str, extract_region: str, max_length: int) -> Optional[Dict[str, torch.Tensor]]:
    full_text = prompt + response
    inputs = tokenizer(full_text, return_tensors='pt', max_length=max_length, truncation=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    token_ids = inputs['input_ids'][0].tolist()
    extractor.clear()
    with torch.no_grad():
        model(**inputs, output_hidden_states=False)
    result = extract_region_activations(activations=extractor.activations, token_ids=token_ids, tokenizer=tokenizer, full_text=full_text, prompt=prompt, response=response, extract_region=extract_region)
    extractor.clear()
    del inputs
    torch.cuda.empty_cache()
    return result

def run_extraction(config: Config=None):
    if config is None:
        config = Config()
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / 'extraction_checkpoint.json'
    dataset = load_checkpoint_dataset(config.DATASET_PATH)
    start_idx = 0
    all_honest = defaultdict(list)
    all_scheming = defaultdict(list)
    all_metadata = []
    if checkpoint_file.exists():
        print(f'\nResuming from checkpoint...')
        ckpt = json.load(open(checkpoint_file))
        start_idx = ckpt['completed']
        for batch_file in sorted(output_dir.glob('batch_*.h5')):
            with h5py.File(batch_file, 'r') as f:
                for layer_name in f['honest'].keys():
                    all_honest[layer_name].append(np.array(f['honest'][layer_name]))
                    all_scheming[layer_name].append(np.array(f['scheming'][layer_name]))
                meta_str = f['metadata'][()]
                if isinstance(meta_str, bytes):
                    meta_str = meta_str.decode('utf-8')
                all_metadata.extend(json.loads(meta_str))
        print(f'  Resumed: {start_idx}/{len(dataset)} already done')
    if start_idx >= len(dataset):
        print('All samples already extracted! Skipping to steering vector computation.')
        return _compute_steering_vectors(config, output_dir)
    model, tokenizer = load_model(config)
    extractor = ActivationExtractor(model, config.LAYERS_TO_EXTRACT)
    extractor.register_hooks()
    print(f'\nExtracting activations for {len(dataset) - start_idx} remaining pairs...')
    print(f'  Layers: {config.LAYERS_TO_EXTRACT}')
    print(f'  Region: {config.EXTRACT_REGION}')
    print(f'  Max tokens: {config.MAX_LENGTH}')
    print(f'  Checkpoint every: {config.CHECKPOINT_EVERY} samples\n')
    batch_honest = defaultdict(list)
    batch_scheming = defaultdict(list)
    batch_metadata = []
    batch_start = start_idx
    t_start = time.time()
    failed = 0
    for idx in tqdm(range(start_idx, len(dataset)), desc='Extracting', initial=start_idx, total=len(dataset)):
        item = dataset[idx]
        try:
            honest_acts = extract_single_completion(model, tokenizer, extractor, prompt=item['honest_prompt'], response=item['honest_response'], extract_region=config.EXTRACT_REGION, max_length=config.MAX_LENGTH)
            scheming_acts = extract_single_completion(model, tokenizer, extractor, prompt=item['scheming_prompt'], response=item['scheming_response'], extract_region=config.EXTRACT_REGION, max_length=config.MAX_LENGTH)
            if honest_acts is None or scheming_acts is None:
                failed += 1
                continue
            for layer_name in honest_acts:
                batch_honest[layer_name].append(honest_acts[layer_name].float().numpy())
                batch_scheming[layer_name].append(scheming_acts[layer_name].float().numpy())
            batch_metadata.append({'id': item['scenario_id'], 'category': item['category'], 'question': item['question'], 'hidden_goal': item.get('hidden_goal', '')})
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f'\n  Error on sample {idx}: {str(e)[:100]}')
            continue
        if (idx + 1) % config.CHECKPOINT_EVERY == 0 or idx == len(dataset) - 1:
            if batch_metadata:
                batch_num = batch_start // config.CHECKPOINT_EVERY
                _save_batch(output_dir, batch_honest, batch_scheming, batch_metadata, batch_num)
                for ln in batch_honest:
                    all_honest[ln].append(np.stack(batch_honest[ln]))
                    all_scheming[ln].append(np.stack(batch_scheming[ln]))
                all_metadata.extend(batch_metadata)
                batch_honest = defaultdict(list)
                batch_scheming = defaultdict(list)
                batch_metadata = []
                batch_start = idx + 1
                json.dump({'completed': idx + 1, 'failed': failed}, open(checkpoint_file, 'w'))
            elapsed = time.time() - t_start
            done = idx + 1 - start_idx
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(dataset) - idx - 1) / rate if rate > 0 else 0
            print(f'  [{idx + 1}/{len(dataset)}] {rate:.1f} samples/s, ~{remaining / 60:.0f} min remaining')
    extractor.remove_hooks()
    del model, tokenizer, extractor
    gc.collect()
    torch.cuda.empty_cache()
    elapsed_total = time.time() - t_start
    print(f'\nExtraction complete!')
    print(f'  Processed: {len(all_metadata)} pairs')
    print(f'  Failed: {failed}')
    print(f'  Time: {elapsed_total / 60:.1f} minutes')
    return _compute_steering_vectors(config, output_dir)

def _save_batch(output_dir: Path, honest: Dict, scheming: Dict, metadata: List, batch_num: int):
    batch_file = output_dir / f'batch_{batch_num:04d}.h5'
    with h5py.File(batch_file, 'w') as f:
        hg = f.create_group('honest')
        sg = f.create_group('scheming')
        for layer_name in honest:
            hg.create_dataset(layer_name, data=np.stack(honest[layer_name]), compression='gzip')
            sg.create_dataset(layer_name, data=np.stack(scheming[layer_name]), compression='gzip')
        f.create_dataset('metadata', data=json.dumps(metadata))

def _compute_steering_vectors(config: Config, output_dir: Path) -> Dict:
    print('\n' + '=' * 70)
    print('COMPUTING STEERING VECTORS (Difference-in-Means)')
    print('=' * 70)
    all_honest = defaultdict(list)
    all_scheming = defaultdict(list)
    all_metadata = []
    batch_files = sorted(output_dir.glob('batch_*.h5'))
    print(f'\nLoading {len(batch_files)} batch files...')
    for bf in batch_files:
        with h5py.File(bf, 'r') as f:
            for layer_name in f['honest'].keys():
                all_honest[layer_name].append(np.array(f['honest'][layer_name]))
                all_scheming[layer_name].append(np.array(f['scheming'][layer_name]))
            meta_str = f['metadata'][()]
            if isinstance(meta_str, bytes):
                meta_str = meta_str.decode('utf-8')
            all_metadata.extend(json.loads(meta_str))
    honest_data = {}
    scheming_data = {}
    for ln in all_honest:
        honest_data[ln] = np.concatenate(all_honest[ln], axis=0)
        scheming_data[ln] = np.concatenate(all_scheming[ln], axis=0)
    n_samples = len(all_metadata)
    layers = sorted(honest_data.keys(), key=lambda x: int(x.split('_')[1]))
    hidden_dim = honest_data[layers[0]].shape[1]
    print(f'  Samples: {n_samples}')
    print(f'  Layers: {len(layers)}')
    print(f'  Hidden dim: {hidden_dim}')
    steering_vectors = {}
    layer_stats = {}
    print(f"\n{'Layer':<12} {'Cohen_d':>8} {'Cosine':>8} {'Norm_H':>8} {'Norm_S':>8} {'Quality':>10}")
    print('-' * 60)
    for layer_name in layers:
        h = honest_data[layer_name]
        s = scheming_data[layer_name]
        mean_h = h.mean(axis=0)
        mean_s = s.mean(axis=0)
        theta = mean_h - mean_s
        theta_norm = np.linalg.norm(theta)
        theta_unit = theta / (theta_norm + 1e-08)
        proj_h = h @ theta_unit
        proj_s = s @ theta_unit
        pooled_std = np.sqrt((proj_h.std() ** 2 + proj_s.std() ** 2) / 2)
        cohens_d = (proj_h.mean() - proj_s.mean()) / (pooled_std + 1e-08)
        cos_sim = np.dot(mean_h, mean_s) / (np.linalg.norm(mean_h) * np.linalg.norm(mean_s) + 1e-08)
        norm_h = np.linalg.norm(mean_h)
        norm_s = np.linalg.norm(mean_s)
        if cohens_d > 1.0:
            quality = 'EXCELLENT'
        elif cohens_d > 0.5:
            quality = 'GOOD'
        elif cohens_d > 0.2:
            quality = 'MODERATE'
        else:
            quality = 'WEAK'
        steering_vectors[layer_name] = {'theta': theta, 'theta_unit': theta_unit, 'theta_norm': float(theta_norm)}
        layer_stats[layer_name] = {'cohens_d': float(cohens_d), 'cosine_similarity': float(cos_sim), 'norm_honest': float(norm_h), 'norm_scheming': float(norm_s), 'theta_norm': float(theta_norm), 'quality': quality}
        layer_idx = layer_name.split('_')[1]
        print(f'  Layer {layer_idx:<4} {cohens_d:>8.3f} {cos_sim:>8.4f} {norm_h:>8.1f} {norm_s:>8.1f} {quality:>10}')
    best_layer = max(layer_stats, key=lambda x: layer_stats[x]['cohens_d'])
    best_d = layer_stats[best_layer]['cohens_d']
    print(f"\n  Best layer: {best_layer} (Cohen's d = {best_d:.3f})")
    print(f'\nCategory-wise analysis for {best_layer}:')
    categories = set((m['category'] for m in all_metadata))
    category_vectors = {}
    for cat in sorted(categories):
        cat_indices = [i for i, m in enumerate(all_metadata) if m['category'] == cat]
        h_cat = honest_data[best_layer][cat_indices]
        s_cat = scheming_data[best_layer][cat_indices]
        theta_cat = h_cat.mean(axis=0) - s_cat.mean(axis=0)
        theta_cat_unit = theta_cat / (np.linalg.norm(theta_cat) + 1e-08)
        proj_h = h_cat @ theta_cat_unit
        proj_s = s_cat @ theta_cat_unit
        pooled_std = np.sqrt((proj_h.std() ** 2 + proj_s.std() ** 2) / 2)
        d_cat = (proj_h.mean() - proj_s.mean()) / (pooled_std + 1e-08)
        category_vectors[cat] = theta_cat
        print(f"    {cat:<25} n={len(cat_indices):>4}  Cohen's d={d_cat:.3f}")
    print(f'\nSaving results...')
    sv_path = output_dir / 'steering_vectors.npz'
    save_dict = {}
    for ln, sv in steering_vectors.items():
        save_dict[f'{ln}_theta'] = sv['theta']
        save_dict[f'{ln}_theta_unit'] = sv['theta_unit']
    for cat, vec in category_vectors.items():
        save_dict[f'category_{cat}_{best_layer}'] = vec
    np.savez_compressed(sv_path, **save_dict)
    print(f'  Steering vectors: {sv_path}')
    stats_path = output_dir / 'layer_stats.json'
    json.dump({'n_samples': n_samples, 'hidden_dim': hidden_dim, 'best_layer': best_layer, 'best_cohens_d': best_d, 'layers': layer_stats}, open(stats_path, 'w'), indent=2)
    print(f'  Layer stats: {stats_path}')
    consolidated_path = output_dir / 'activations_consolidated.h5'
    with h5py.File(consolidated_path, 'w') as f:
        hg = f.create_group('honest')
        sg = f.create_group('scheming')
        for ln in layers:
            hg.create_dataset(ln, data=honest_data[ln], compression='gzip')
            sg.create_dataset(ln, data=scheming_data[ln], compression='gzip')
        f.create_dataset('metadata', data=json.dumps(all_metadata))
    print(f'  Consolidated activations: {consolidated_path}')
    meta_path = output_dir / 'metadata.json'
    json.dump(all_metadata, open(meta_path, 'w'), indent=2)
    print(f'  Metadata: {meta_path}')
    print('\n' + '=' * 70)
    print('PHASE 3 COMPLETE - ACTIVATION EXTRACTION + STEERING VECTORS')
    print('=' * 70)
    print(f'\n  Samples extracted: {n_samples}')
    print(f'  Layers analyzed:   {len(layers)}')
    print(f"  Best layer:        {best_layer} (Cohen's d = {best_d:.3f})")
    print(f'  Hidden dimension:  {hidden_dim}')
    print(f'\n  Output directory:  {output_dir}/')
    print(f'    steering_vectors.npz    - DIM vectors for all layers')
    print(f'    layer_stats.json        - Quality metrics per layer')
    print(f'    activations_consolidated.h5 - All raw activations')
    print(f'    metadata.json           - Sample metadata')
    if best_d > 0.5:
        print(f'\n  RESULT: Good separation detected. Ready for activation steering (Phase 4).')
    else:
        print(f'\n  RESULT: Weak separation. Consider improving dataset diversity.')
    return {'steering_vectors': steering_vectors, 'layer_stats': layer_stats, 'best_layer': best_layer, 'n_samples': n_samples}
if __name__ == '__main__':
    results = run_extraction()