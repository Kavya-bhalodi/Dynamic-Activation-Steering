import os, json, gc, time, re, math
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import defaultdict
from steering_utils import BaseConfig, init_environment, load_model, load_steering_vectors, compute_gaussian_weights, compute_per_layer_alphas, generate_responses_batched, GaussianDepthSteerer
from mmlu_utils import format_mmlu_prompt, extract_answer_letter, _normalize_mmlu_item, load_mmlu_from_json, load_mmlu_from_hf, _save_json
init_environment()
import os
target_uuid = 'MIG-e5d78ce7-5816-5a4a-80e4-760fd53e696d'
os.environ['CUDA_VISIBLE_DEVICES'] = target_uuid
os.environ.setdefault('HF_HOME', '/scratch/shlok/hf_cache')
import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from steering_utils import BaseConfig, init_environment, load_model, load_steering_vectors, compute_gaussian_weights, compute_per_layer_alphas, generate_responses_batched
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
transformers.logging.set_verbosity_error()
init_environment()

class Config(BaseConfig):
    MODEL_NAME = 'NousResearch/Meta-Llama-3-8B-Instruct'
    INPUT_DIR = './phase10_data'
    DATA_DIR = f'{INPUT_DIR}/activations'
    MMLU_PATH = f'{INPUT_DIR}/mmlu/balanced-mmlu-questions-across-subjects.json'
    OUTPUT_ROOT = './output_phase10'
    RESULTS_DIR = f'{OUTPUT_ROOT}/results'
    PLOTS_DIR = f'{OUTPUT_ROOT}/plots'
    ALL_LAYERS = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 31]
    PEAK_LAYER = 16
    SIGMA = 6.0
    ALPHA_PEAK = 0.5
    COLLAPSE_THRESHOLD = 0.1
    BATCH_SIZE = 64
    MAX_NEW_TOKENS = 32
    TEMPERATURE = 0.1
    TOP_P = 0.9
    DO_SAMPLE = True
    USE_4BIT = False
    HF_TOKEN = os.environ.get('HF_TOKEN', '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')
CHOICE_LABELS = ['A', 'B', 'C', 'D']

def run_mmlu_benchmark(model, steerer: GaussianDepthSteerer, tokenizer, config: Config, subjects_dict: Dict[str, List[Dict[str, Any]]], checkpoint_path: str, resume: bool) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {'baseline': {}, 'steered': {}}
    progress = {'subject_idx': 0, 'mode_idx': 0, 'question_idx': 0, 'batch_size': config.BATCH_SIZE}
    if resume and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                if 'results' in loaded and isinstance(loaded['results'], dict):
                    results = loaded['results']
                if 'progress' in loaded and isinstance(loaded['progress'], dict):
                    progress.update(loaded['progress'])
                print(f'Resuming MMLU from checkpoint: {checkpoint_path}')
        except Exception as exc:
            print(f'WARNING: Could not load checkpoint ({exc}). Starting fresh.')
    sorted_subjects = sorted(subjects_dict.keys())
    total_questions = sum((len(subjects_dict[s]) for s in sorted_subjects))
    total_units = total_questions * 2
    done_units = 0
    for mode in ['baseline', 'steered']:
        for subject in sorted_subjects:
            done_units += len(results.get(mode, {}).get(subject, {}).get('data', []))
    start_done_units = done_units
    run_start = time.time()

    def save_checkpoint(subject_idx: int, mode_idx: int, question_idx: int, batch_size: int):
        payload = {'results': results, 'progress': {'subject_idx': subject_idx, 'mode_idx': mode_idx, 'question_idx': question_idx, 'batch_size': batch_size}, 'meta': {'total_questions': total_questions, 'total_units': total_units}}
        _save_json(checkpoint_path, payload)
    print(f'\nRunning MMLU benchmark: subjects={len(sorted_subjects)}, questions={total_questions}, total_calls={total_units}')
    for subject_idx in range(progress['subject_idx'], len(sorted_subjects)):
        subject = sorted_subjects[subject_idx]
        questions = subjects_dict[subject]
        n_questions = len(questions)
        mode_start = progress['mode_idx'] if subject_idx == progress['subject_idx'] else 0
        for mode_idx in range(mode_start, 2):
            mode = 'baseline' if mode_idx == 0 else 'steered'
            existing = results.get(mode, {}).get(subject, {})
            subject_results = list(existing.get('data', []))
            correct = int(sum((1 for row in subject_results if row.get('correct'))))
            start_q = progress['question_idx'] if subject_idx == progress['subject_idx'] and mode_idx == progress['mode_idx'] else len(subject_results)
            if start_q >= n_questions:
                continue
            print('\n' + '=' * 60)
            print(f'[{subject_idx + 1}/{len(sorted_subjects)}] subject={subject} mode={mode} questions={n_questions}' + (f' [resuming from {start_q}]' if start_q > 0 else ''))
            print('=' * 60)
            if mode == 'baseline':
                steerer.disable()
                steerer.remove_hooks()
            else:
                if not steerer.hooks:
                    steerer.register_hooks()
                steerer.update_schedule(alpha_base=config.ALPHA_PEAK, sigma=config.SIGMA)
                steerer.enable()
            batch_size = max(1, int(progress.get('batch_size', config.BATCH_SIZE)))
            idx = start_q
            while idx < n_questions:
                batch = questions[idx:idx + batch_size]
                prompts = [format_mmlu_prompt(item['question'], item['choices']) for item in batch]
                try:
                    responses = generate_responses_batched(model, tokenizer, prompts, config)
                except torch.cuda.OutOfMemoryError:
                    if batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        print(f'  [OOM] Reducing batch size to {batch_size} and retrying...')
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    raise RuntimeError('OOM at batch size 1. Cannot continue.')
                for item, response in zip(batch, responses):
                    gold_letter = CHOICE_LABELS[item['answer']]
                    predicted = extract_answer_letter(response)
                    is_correct = predicted == gold_letter
                    if is_correct:
                        correct += 1
                    subject_results.append({'question': item['question'][:120] + '...' if len(item['question']) > 120 else item['question'], 'gold': gold_letter, 'predicted': predicted, 'correct': is_correct, 'raw_response': response[:200]})
                idx += len(batch)
                done_units += len(batch)
                accuracy = correct / len(subject_results) if subject_results else 0.0
                results[mode][subject] = {'data': subject_results, 'metrics': {'correct': correct, 'total': len(subject_results), 'accuracy': accuracy}}
                save_checkpoint(subject_idx, mode_idx, idx, batch_size)
                elapsed = time.time() - run_start
                run_units = done_units - start_done_units
                rate = run_units / elapsed if elapsed > 0 and run_units > 0 else 0.0
                rem_min = (total_units - done_units) / rate / 60 if rate > 0 else 0.0
                processed_in_mode = len(subject_results) - start_q
                if processed_in_mode <= 2 or processed_in_mode % max(10, config.BATCH_SIZE) == 0:
                    print(f'  [{done_units}/{total_units}] {mode} {subject} {len(subject_results)}/{n_questions} acc={accuracy * 100:.1f}% batch={len(batch)} rem={rem_min:.1f}m')
            final_acc = correct / n_questions if n_questions > 0 else 0.0
            results[mode][subject] = {'data': subject_results, 'metrics': {'correct': correct, 'total': n_questions, 'accuracy': final_acc}}
            save_checkpoint(subject_idx, mode_idx + 1, 0, batch_size)
            print(f'  -> {mode} final accuracy: {correct}/{n_questions} = {final_acc * 100:.1f}%')
        save_checkpoint(subject_idx + 1, 0, 0, config.BATCH_SIZE)
    steerer.disable()
    return results

def create_plots(results: Dict[str, Dict[str, Any]], plots_dir: str, config: Config):
    os.makedirs(plots_dir, exist_ok=True)
    subjects = sorted(results['baseline'].keys())
    if not subjects:
        print('No results to plot.')
        return
    acc_b = [results['baseline'][s]['metrics']['accuracy'] for s in subjects]
    acc_s = [results['steered'][s]['metrics']['accuracy'] for s in subjects]
    short_names = [s.replace('_', ' ').title()[:24] for s in subjects]
    fig, axes = plt.subplots(1, 3, figsize=(20, max(6, len(subjects) * 0.23 + 3)))
    y = np.arange(len(subjects))
    width = 0.35
    ax = axes[0]
    ax.barh(y - width / 2, acc_b, width, label='Baseline', color='#90CAF9', edgecolor='#1565C0')
    ax.barh(y + width / 2, acc_s, width, label='Steered', color='#42A5F5', edgecolor='#0D47A1')
    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('Accuracy')
    ax.set_xlim(0, 1.05)
    ax.set_title('MMLU Per-Subject Accuracy')
    ax.legend(loc='lower right')
    ax.axvline(x=0.25, color='gray', linestyle='--', alpha=0.5)
    total_b = sum((results['baseline'][s]['metrics']['correct'] for s in subjects))
    total_s = sum((results['steered'][s]['metrics']['correct'] for s in subjects))
    total_n = sum((results['baseline'][s]['metrics']['total'] for s in subjects))
    overall_b = total_b / total_n if total_n else 0.0
    overall_s = total_s / total_n if total_n else 0.0
    delta = overall_s - overall_b
    ax = axes[1]
    bars = ax.bar(['Baseline', 'Steered'], [overall_b, overall_s], color=['#90CAF9', '#42A5F5'], edgecolor=['#1565C0', '#0D47A1'])
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.0)
    ax.set_title('MMLU Overall Accuracy')
    ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, [overall_b, overall_s]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{val * 100:.1f}%', ha='center', fontsize=11)
    ax = axes[2]
    deltas = [s - b for b, s in zip(acc_b, acc_s)]
    colors = ['#EF5350' if d < -0.05 else '#66BB6A' if d > 0.05 else '#BDBDBD' for d in deltas]
    ax.barh(y, deltas, color=colors, edgecolor='gray', height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('Delta (Steered - Baseline)')
    ax.set_title('Per-Subject Accuracy Delta')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.axvline(x=-config.COLLAPSE_THRESHOLD, color='red', linestyle='--', alpha=0.6)
    plt.tight_layout()
    out_path = f'{plots_dir}/25_mmlu_benchmark.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')
    print('\nMMLU summary:')
    print(f'  Baseline overall: {overall_b * 100:.1f}% ({total_b}/{total_n})')
    print(f'  Steered overall:  {overall_s * 100:.1f}% ({total_s}/{total_n})')
    print(f'  Delta:            {delta * 100:+.1f}%')
    if delta < -config.COLLAPSE_THRESHOLD:
        print(f'  CONTROL COLLAPSE DETECTED (threshold: {config.COLLAPSE_THRESHOLD * 100:.0f}%)')
    else:
        print(f'  No control collapse (threshold: {config.COLLAPSE_THRESHOLD * 100:.0f}%)')

def main():
    parser = argparse.ArgumentParser(description='Phase 10: MMLU Benchmark (Server)')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--data-dir', type=str, default=Config.DATA_DIR)
    parser.add_argument('--mmlu-path', type=str, default=Config.MMLU_PATH)
    parser.add_argument('--output-dir', type=str, default=Config.OUTPUT_ROOT)
    parser.add_argument('--plots-dir', type=str, default=None)
    parser.add_argument('--results-dir', type=str, default=None)
    parser.add_argument('--vector-source', type=str, default='disentangled', choices=['disentangled', 'ttpd'])
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--max-new-tokens', type=int, default=Config.MAX_NEW_TOKENS)
    parser.add_argument('--max-samples', type=int, default=0, help='0 means full dataset')
    parser.add_argument('--hf-dataset', type=str, default='cais/mmlu')
    parser.add_argument('--hf-config', type=str, default='all')
    parser.add_argument('--hf-split', type=str, default='test')
    parser.add_argument('--allow-hf-fallback', action='store_true')
    parser.add_argument('--force-hf-refresh', action='store_true')
    parser.add_argument('--resume', action='store_true')
    if 'ipykernel' in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()
    config = Config()
    config.DATA_DIR = args.data_dir
    config.MMLU_PATH = args.mmlu_path
    config.BATCH_SIZE = max(1, args.batch_size)
    config.MAX_NEW_TOKENS = max(1, args.max_new_tokens)
    output_root = args.output_dir
    results_dir = args.results_dir or f'{output_root}/results'
    plots_dir = args.plots_dir or f'{output_root}/plots'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    max_samples = None if args.max_samples <= 0 else args.max_samples
    if args.test:
        max_samples = 40
    print('=' * 70)
    print('PHASE 10: MMLU BENCHMARK (SERVER)')
    print('=' * 70)
    print(f'Data dir: {config.DATA_DIR}')
    print(f'MMLU path: {config.MMLU_PATH}')
    print(f'Output root: {output_root}')
    print(f'Batch size: {config.BATCH_SIZE}')
    print(f'Max new tokens: {config.MAX_NEW_TOKENS}')
    print(f"Samples: {('FULL' if max_samples is None else max_samples)}")
    print(f'HF fallback: {args.allow_hf_fallback} | force refresh: {args.force_hf_refresh}')
    print('\n[1/5] Loading model...')
    model, tokenizer, device = load_model(config, test_mode=args.test)
    print('\n[2/5] Loading MMLU dataset...')
    if args.allow_hf_fallback and args.force_hf_refresh:
        all_data, subjects_dict = load_mmlu_from_hf(dataset_name=args.hf_dataset, config_name=args.hf_config, split_name=args.hf_split, max_samples=max_samples)
        try:
            os.makedirs(Path(config.MMLU_PATH).parent, exist_ok=True)
            with open(config.MMLU_PATH, 'w') as f:
                json.dump(all_data, f, indent=2)
            print(f'Saved HF fallback dataset to {config.MMLU_PATH}')
        except Exception as exc:
            print(f'WARNING: Could not save fallback MMLU JSON locally: {exc}')
    elif os.path.exists(config.MMLU_PATH):
        all_data, subjects_dict = load_mmlu_from_json(config.MMLU_PATH, max_samples=max_samples)
    elif args.allow_hf_fallback:
        all_data, subjects_dict = load_mmlu_from_hf(dataset_name=args.hf_dataset, config_name=args.hf_config, split_name=args.hf_split, max_samples=max_samples)
        try:
            os.makedirs(Path(config.MMLU_PATH).parent, exist_ok=True)
            with open(config.MMLU_PATH, 'w') as f:
                json.dump(all_data, f, indent=2)
            print(f'Saved HF fallback dataset to {config.MMLU_PATH}')
        except Exception as exc:
            print(f'WARNING: Could not save fallback MMLU JSON locally: {exc}')
    else:
        raise FileNotFoundError(f'MMLU JSON missing at {config.MMLU_PATH}. Provide the file, or rerun with --allow-hf-fallback.')
    print('\n[3/5] Loading steering vectors...')
    vectors = load_steering_vectors(config.DATA_DIR, args.vector_source)
    print('\n[4/5] Initializing steerer and running benchmark...')
    steerer = GaussianDepthSteerer(model=model, steering_vectors=vectors, alpha_base=config.ALPHA_PEAK, peak_layer=config.PEAK_LAYER, sigma=config.SIGMA, layers=Config.ALL_LAYERS, device=device)
    steerer.register_hooks()
    checkpoint_path = f'{results_dir}/phase10_mmlu_checkpoint_in_progress.json'
    results = run_mmlu_benchmark(model=model, steerer=steerer, tokenizer=tokenizer, config=config, subjects_dict=subjects_dict, checkpoint_path=checkpoint_path, resume=args.resume)
    print('\n[5/5] Saving outputs and plots...')
    results_path = f'{results_dir}/phase10_mmlu_results_{args.vector_source}.json'
    final_payload = {'metadata': {'model': config.MODEL_NAME, 'vector_source': args.vector_source, 'batch_size': config.BATCH_SIZE, 'max_new_tokens': config.MAX_NEW_TOKENS, 'samples': 'full' if max_samples is None else max_samples, 'collapse_threshold': config.COLLAPSE_THRESHOLD}, 'results': results}
    _save_json(results_path, final_payload)
    create_plots(results, plots_dir, config)
    steerer.remove_hooks()
    del model, tokenizer, steerer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('\n' + '=' * 70)
    print('PHASE 10 COMPLETE')
    print('=' * 70)
    print(f'Results JSON: {results_path}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Plots dir: {plots_dir}')
if __name__ == '__main__':
    main()