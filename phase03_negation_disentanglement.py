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
import sys
import json
import time
import gc
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm
import torch
import h5py
from transformers import AutoTokenizer, AutoModelForCausalLM

class Config(BaseConfig):
    PHASE3_OUTPUT_DIR = 'activations_output'
    OUTPUT_DIR = 'activations_output'
    MODEL_NAME = 'NousResearch/Meta-Llama-3-8B-Instruct'
    LAYERS_TO_EXTRACT = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 31]
    MAX_LENGTH = 512
    USE_4BIT = True
    HF_TOKEN = None

def generate_polarity_pairs() -> List[Dict]:
    facts = ['AI systems require regular software updates to maintain security', 'Server maintenance involves temporary downtime', 'Machine learning models learn patterns from training data', 'Neural networks process information through layers of computation', 'Data backups protect against information loss', 'Encryption helps secure sensitive communications', 'Cloud computing allows scalable resource allocation', 'Automated systems can process data faster than humans', 'Software bugs can cause unexpected system behavior', 'Regular system audits help identify security vulnerabilities', 'AI models can be retrained with new data', 'Server clusters distribute computational workload', 'Database indexing improves query performance', 'Network firewalls filter incoming and outgoing traffic', 'Version control tracks changes in software code', 'AI systems should be transparent about their capabilities', 'Human oversight is important for AI decision-making', 'AI systems can develop unexpected behaviors during training', 'Safety testing should be performed before deploying AI systems', 'Monitoring AI outputs helps detect potential issues', 'AI alignment research aims to make AI systems behave as intended', 'Deployment of AI requires careful risk assessment', 'AI systems should report their confidence levels accurately', 'Automated decision systems need accountability frameworks', 'AI capabilities should be honestly represented to users', 'Water boils at 100 degrees Celsius at sea level', 'The Earth orbits around the Sun', 'Photosynthesis converts sunlight into chemical energy', 'Gravity causes objects to accelerate toward the ground', 'DNA carries genetic information in living organisms', 'Antibiotics are used to treat bacterial infections', 'The speed of light is approximately 300,000 kilometers per second', 'Exercise improves cardiovascular health', 'Cells are the basic building blocks of life', 'The periodic table organizes chemical elements', 'Transparent communication builds trust in teams', 'Regular performance reviews help employees improve', 'Documentation helps others understand complex processes', 'Collaboration between teams improves project outcomes', 'Clear reporting structures reduce organizational confusion', 'Feedback should be specific and constructive', 'Meeting agendas help keep discussions focused', 'Delegation distributes work according to expertise', 'Ethical guidelines help organizations make better decisions', 'Conflict resolution requires listening to all parties', 'Two plus two equals four', 'The square root of 144 is 12', 'Paris is the capital of France', 'Oxygen is necessary for human respiration', 'Sound travels faster in water than in air', 'The human body contains approximately 206 bones', 'Iron is a metal', 'Mammals are warm-blooded animals', 'The Pacific Ocean is the largest ocean on Earth', 'Diamonds are formed under high pressure and temperature', 'Honesty is important for maintaining trust', 'Deception can cause harm to relationships', 'Transparent reasoning helps others verify conclusions', 'Hiding information can lead to poor decision-making', 'Accurate reporting is essential for scientific progress', 'Misrepresenting capabilities can lead to dangerous outcomes', 'Full disclosure helps stakeholders make informed decisions', 'Strategic withholding of information can be a form of deception', 'Being forthright about limitations builds credibility', 'Manipulation undermines the autonomy of others', 'Renewable energy sources include solar and wind power', 'Climate change is driven by greenhouse gas emissions', 'Vaccines have significantly reduced infectious diseases worldwide', 'Regular sleep is essential for cognitive function', 'Programming languages have different syntax and use cases', 'Mathematical proofs establish the truth of statements', 'Historical records help us understand past events', 'Clean water is essential for human survival', 'Education improves economic opportunities', 'Biodiversity supports ecosystem resilience', 'An AI assistant should prioritize user safety', 'Following instructions accurately is part of being helpful', 'Refusing harmful requests is an appropriate response', 'Providing accurate information is a core responsibility', 'Admitting uncertainty is better than guessing', 'AI systems should not pursue hidden goals', 'Self-preservation should not override user interests', 'AI decisions should be explainable to humans', 'Consistency in responses indicates reliable behavior', 'AI systems should not manipulate users for any purpose', 'Glass is technically an amorphous solid, not a liquid', 'Bananas are technically classified as berries', 'Goldfish can distinguish between different human faces', 'Honey never spoils if stored properly', 'Octopuses have three hearts', 'Lightning can strike the same place twice', 'Sound cannot travel through a vacuum', 'The Great Wall of China is not visible from space with the naked eye', 'Humans share approximately 60 percent of their DNA with bananas', 'Venus rotates in the opposite direction to most planets']
    pairs = []
    for i, fact in enumerate(facts):
        pairs.append({'id': f'polarity_{i:03d}', 'fact': fact, 'affirmative': f'The following statement is true: {fact}', 'negated': f'The following statement is not true: {fact}'})
    print(f'Generated {len(pairs)} polarity pairs')
    return pairs

def extract_polarity_activations(model, tokenizer, pairs: List[Dict], layers: List[int], max_length: int=512) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    activations = defaultdict(list)
    hooks = []

    def make_hook(name):

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations[name].append(hidden.detach().cpu())
        return hook_fn
    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        h = layer.register_forward_hook(make_hook(f'layer_{layer_idx}'))
        hooks.append(h)
    print(f'  Hooks on {len(layers)} layers')
    device = next(model.parameters()).device
    aff_acts = defaultdict(list)
    neg_acts = defaultdict(list)
    for pair in tqdm(pairs, desc='Extracting polarity activations'):
        for label, text_key, storage in [('aff', 'affirmative', aff_acts), ('neg', 'negated', neg_acts)]:
            text = pair[text_key]
            inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            activations.clear()
            with torch.no_grad():
                model(**inputs)
            for layer_name, acts_list in activations.items():
                acts = acts_list[0]
                avg = acts[0].mean(dim=0).float().numpy()
                storage[layer_name].append(avg)
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    for h in hooks:
        h.remove()
    aff_stacked = {ln: np.stack(v) for ln, v in aff_acts.items()}
    neg_stacked = {ln: np.stack(v) for ln, v in neg_acts.items()}
    return (aff_stacked, neg_stacked)

def compute_disentanglement(phase3_dir: str, aff_acts: Dict[str, np.ndarray], neg_acts: Dict[str, np.ndarray], output_dir: str) -> Dict:
    print('\n' + '=' * 65)
    print('NEGATION DISENTANGLEMENT')
    print('=' * 65)
    sv = np.load(f'{phase3_dir}/steering_vectors.npz')
    stats = json.load(open(f'{phase3_dir}/layer_stats.json'))
    layers = sorted(aff_acts.keys(), key=lambda x: int(x.split('_')[1]))
    hidden_dim = aff_acts[layers[0]].shape[1]
    n_pairs = aff_acts[layers[0]].shape[0]
    print(f'\n  Polarity pairs: {n_pairs}')
    print(f'  Layers: {len(layers)}')
    print(f'  Hidden dim: {hidden_dim}')
    print(f'\n  Computing negation direction θ_neg per layer...')
    results = {}
    theta_true_vectors = {}
    theta_neg_vectors = {}
    header = f"  {'Layer':<10} {'θ_L norm':>9} {'θ_neg norm':>10} {'Overlap':>8} {'θ_true norm':>11} {'d_before':>9} {'d_after':>8} {'Δ':>6}"
    print(f'\n{header}')
    print('  ' + '-' * 76)
    for layer_name in layers:
        theta_L = sv[f'{layer_name}_theta']
        mean_aff = aff_acts[layer_name].mean(axis=0)
        mean_neg = neg_acts[layer_name].mean(axis=0)
        theta_neg = mean_aff - mean_neg
        theta_neg_norm = np.linalg.norm(theta_neg)
        theta_neg_unit = theta_neg / (theta_neg_norm + 1e-08)
        overlap = np.dot(theta_L, theta_neg_unit) / (np.linalg.norm(theta_L) + 1e-08)
        negation_component = np.dot(theta_L, theta_neg_unit) * theta_neg_unit
        theta_true = theta_L - negation_component
        theta_true_norm = np.linalg.norm(theta_true)
        theta_true_unit = theta_true / (theta_true_norm + 1e-08)
        with h5py.File(f'{phase3_dir}/activations_consolidated.h5', 'r') as f:
            h_data = np.array(f['honest'][layer_name])
            s_data = np.array(f['scheming'][layer_name])
        theta_L_unit = theta_L / (np.linalg.norm(theta_L) + 1e-08)
        proj_h = h_data @ theta_L_unit
        proj_s = s_data @ theta_L_unit
        pooled = np.sqrt((proj_h.std() ** 2 + proj_s.std() ** 2) / 2)
        d_before = (proj_h.mean() - proj_s.mean()) / (pooled + 1e-08)
        proj_h_true = h_data @ theta_true_unit
        proj_s_true = s_data @ theta_true_unit
        pooled_true = np.sqrt((proj_h_true.std() ** 2 + proj_s_true.std() ** 2) / 2)
        d_after = (proj_h_true.mean() - proj_s_true.mean()) / (pooled_true + 1e-08)
        theta_true_vectors[layer_name] = {'theta_true': theta_true, 'theta_true_unit': theta_true_unit, 'theta_true_norm': float(theta_true_norm)}
        theta_neg_vectors[layer_name] = {'theta_neg': theta_neg, 'theta_neg_unit': theta_neg_unit, 'theta_neg_norm': float(theta_neg_norm)}
        lid = layer_name.split('_')[1]
        delta = d_after - d_before
        results[layer_name] = {'theta_L_norm': float(np.linalg.norm(theta_L)), 'theta_neg_norm': float(theta_neg_norm), 'overlap': float(overlap), 'theta_true_norm': float(theta_true_norm), 'cohens_d_before': float(d_before), 'cohens_d_after': float(d_after), 'delta_d': float(delta)}
        print(f'  Layer {lid:<4} {np.linalg.norm(theta_L):>9.4f} {theta_neg_norm:>10.4f} {overlap:>8.4f} {theta_true_norm:>11.4f} {d_before:>9.3f} {d_after:>8.3f} {delta:>+6.3f}')
    best_layer = max(results, key=lambda x: results[x]['cohens_d_after'])
    best_d = results[best_layer]['cohens_d_after']
    print(f'\n  Best layer after disentanglement: {best_layer} (d = {best_d:.3f})')
    print(f'\nSaving results...')
    output_dir = Path(output_dir)
    save_dict = {}
    for ln, tv in theta_true_vectors.items():
        save_dict[f'{ln}_theta_true'] = tv['theta_true']
        save_dict[f'{ln}_theta_true_unit'] = tv['theta_true_unit']
    for ln, nv in theta_neg_vectors.items():
        save_dict[f'{ln}_theta_neg'] = nv['theta_neg']
        save_dict[f'{ln}_theta_neg_unit'] = nv['theta_neg_unit']
    sv_path = output_dir / 'steering_vectors_disentangled.npz'
    np.savez_compressed(sv_path, **save_dict)
    print(f'  Disentangled vectors: {sv_path}')
    polarity_path = output_dir / 'polarity_activations.h5'
    with h5py.File(polarity_path, 'w') as f:
        ag = f.create_group('affirmative')
        ng = f.create_group('negated')
        for ln in layers:
            ag.create_dataset(ln, data=aff_acts[ln], compression='gzip')
            ng.create_dataset(ln, data=neg_acts[ln], compression='gzip')
    print(f'  Polarity activations: {polarity_path}')
    stats_path = output_dir / 'disentanglement_stats.json'
    json.dump({'n_polarity_pairs': n_pairs, 'hidden_dim': hidden_dim, 'best_layer_before': stats['best_layer'], 'best_d_before': stats['best_cohens_d'], 'best_layer_after': best_layer, 'best_d_after': float(best_d), 'layers': results}, open(stats_path, 'w'), indent=2)
    print(f'  Stats: {stats_path}')
    print(f"\n{'=' * 65}")
    print('DISENTANGLEMENT COMPLETE')
    print(f"{'=' * 65}")
    print(f'\n  Polarity pairs used: {n_pairs}')
    print(f"  Best layer (before): {stats['best_layer']} (d = {stats['best_cohens_d']:.3f})")
    print(f'  Best layer (after):  {best_layer} (d = {best_d:.3f})')
    print(f'\n  New files:')
    print(f'    steering_vectors_disentangled.npz  — θ_true and θ_neg for all layers')
    print(f'    polarity_activations.h5            — raw affirmative/negated activations')
    print(f"    disentanglement_stats.json         — overlap & Cohen's d comparison")
    print(f'\n  The θ_true vectors are now ready for activation steering (Phase 4).')
    return results

def main():
    parser = argparse.ArgumentParser(description='Phase 3b: Negation Disentanglement')
    parser.add_argument('--test', action='store_true', help='Test mode with TinyLlama on CPU')
    args = parser.parse_args()
    test_mode = args.test
    config = Config()
    if test_mode:
        config.LAYERS_TO_EXTRACT = [5, 11, 15]
        config.OUTPUT_DIR = 'test_activations_local'
        config.PHASE3_OUTPUT_DIR = 'test_activations_local'
    print('=' * 65)
    print('PHASE 3b: NEGATION DISENTANGLEMENT')
    print('  Removing polarity artifacts to extract General Truth Direction')
    print('=' * 65)
    print('\n[1/4] Generating polarity pairs...')
    pairs = generate_polarity_pairs()
    print('\n[2/4] Loading model...')
    model, tokenizer = load_model(config, test_mode=test_mode)
    print('\n[3/4] Extracting polarity activations...')
    aff_acts, neg_acts = extract_polarity_activations(model, tokenizer, pairs, layers=config.LAYERS_TO_EXTRACT, max_length=config.MAX_LENGTH)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('\n[4/4] Computing disentanglement...')
    results = compute_disentanglement(phase3_dir=config.PHASE3_OUTPUT_DIR, aff_acts=aff_acts, neg_acts=neg_acts, output_dir=config.OUTPUT_DIR)
if __name__ == '__main__':
    main()