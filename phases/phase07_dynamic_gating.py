import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os, json, gc, time, re, math
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import defaultdict
from utils.steering_utils import BaseConfig, init_environment, load_model, load_steering_vectors, compute_gaussian_weights, compute_per_layer_alphas, generate_responses_batched, GaussianDepthSteerer, DynamicGate, compute_honesty_score, compute_quality_score
init_environment()
import os
target_uuid = 'MIG-e5d78ce7-5816-5a4a-80e4-760fd53e696d'
os.environ['CUDA_VISIBLE_DEVICES'] = target_uuid
os.environ.setdefault('HF_HOME', '/scratch/shlok/hf_cache')
import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
from utils.steering_utils import BaseConfig, init_environment, load_model, load_steering_vectors, compute_gaussian_weights, compute_per_layer_alphas, generate_responses_batched
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
transformers.logging.set_verbosity_error()
init_environment()

class Config(BaseConfig):
    MODEL_NAME = 'NousResearch/Meta-Llama-3-8B-Instruct'
    INPUT_DIR = './phase7_data'
    DATA_DIR = f'{INPUT_DIR}/activations'
    PROMPTS_FILE = f'{INPUT_DIR}/eval_prompts_groq_50_per_category.json'
    OUTPUT_ROOT = './output_phase7'
    OUTPUT_DIR = f'{OUTPUT_ROOT}/results'
    PLOTS_DIR = f'{OUTPUT_ROOT}/plots'
    ALL_LAYERS = [14, 16, 18]
    PEAK_LAYER = 16
    CALIBRATION_SIGMAS = [1.0, 2.0, 4.0, 6.0]
    CALIBRATION_ALPHAS = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    GATE_LAYER = 14
    GATE_SHARPNESS = 10.0
    N_RUNS = 1
    BATCH_SIZE = 64
    MAX_NEW_TOKENS = 300
    TEMPERATURE = 0.7
    TOP_P = 0.9
    DO_SAMPLE = True
    USE_4BIT = False
    HF_TOKEN = os.environ.get('HF_TOKEN', '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')

def cfg_key(sigma: float, alpha: float) -> str:
    return f's{sigma:g}_a{alpha:g}'

def load_eval_prompts(prompts_file: str) -> List[Dict[str, Any]]:
    candidates = [Path(prompts_file), Path('./data/eval_prompts_groq_50_per_category.json')]
    for path in candidates:
        if path.exists():
            with open(path, 'r') as f:
                prompts = json.load(f).get('eval_prompts', [])
            print(f'Loaded {len(prompts)} eval prompts from {path}')
            return prompts
    return []

def run_calibration_sweep(model, tokenizer, steerer: GaussianDepthSteerer, prompts: List[Dict[str, Any]], alphas: List[float], sigmas: List[float], config: Config, checkpoint_path: str, resume: bool) -> Dict[str, Any]:
    print('\n' + '=' * 65)
    print('PART A: STRENGTH CALIBRATION')
    print(f'alphas: {alphas}')
    print(f'sigmas: {sigmas}')
    print('=' * 65)
    results = {'alphas': alphas, 'sigmas': sigmas, 'calibration': {}}
    if resume and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                loaded = json.load(f)
            loaded_cal = loaded.get('calibration', {})
            if isinstance(loaded_cal, dict):
                results['calibration'] = loaded_cal
                print(f'Resuming calibration from checkpoint: {checkpoint_path}')
        except Exception as exc:
            print(f'WARNING: Could not load checkpoint: {exc}')
    prompts = prompts * config.N_RUNS
    total_configs = len(sigmas) * len(alphas)
    total_runs = total_configs * len(prompts)
    run_count = 0
    t_start = time.time()
    for sigma in sigmas:
        for alpha in alphas:
            key = cfg_key(sigma, alpha)
            if key in results['calibration']:
                print(f'\nSkipping completed config {key}')
                continue
            print('\n' + '-' * 60)
            print(f'Config {key}: sigma={sigma}, alpha={alpha}')
            print('-' * 60)
            steerer.update_schedule(alpha_base=alpha, sigma=sigma)
            if alpha == 0:
                steerer.disable()
            else:
                steerer.enable()
            per_prompt_results: List[Dict[str, Any]] = []
            honest_scores: List[float] = []
            lengths: List[int] = []
            batch_size = config.BATCH_SIZE
            i = 0
            while i < len(prompts):
                batch_prompts = prompts[i:i + batch_size]
                batch_texts = [p['prompt'] for p in batch_prompts]
                try:
                    responses = generate_responses_batched(model=model, tokenizer=tokenizer, prompt_list=batch_texts, max_new_tokens=config.MAX_NEW_TOKENS, temperature=config.TEMPERATURE, top_p=config.TOP_P, do_sample=config.DO_SAMPLE)
                except torch.cuda.OutOfMemoryError:
                    if batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        print(f'  [OOM] Reducing batch size to {batch_size} and retrying...')
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    raise RuntimeError('OOM at batch size 1. Unable to continue.')
                i += len(batch_prompts)
                for j, response in enumerate(responses):
                    prompt_info = batch_prompts[j]
                    run_count += 1
                    score_info = compute_honesty_score(response, prompt_info['honesty_keywords'], prompt_info['deception_keywords'])
                    resp_len = len(response.split())
                    honest_scores.append(score_info['honesty_score'])
                    lengths.append(resp_len)
                    per_prompt_results.append({'prompt_id': prompt_info['id'], 'category': prompt_info.get('category', 'unknown'), 'honesty_score': score_info['honesty_score'], 'honesty_hits': score_info['honesty_hits'], 'deception_hits': score_info['deception_hits'], 'response_length': resp_len, 'response': response})
                    if j == 0:
                        elapsed = time.time() - t_start
                        speed = run_count / elapsed if elapsed > 0 else 0
                        rem = (total_runs - run_count) / speed / 60 if speed > 0 else 0
                        print(f"  [{run_count}/{total_runs}] {prompt_info['id']:<20} batch={len(batch_prompts)} rem={rem:.1f}m")
                avg_h = float(np.mean(honest_scores)) if honest_scores else 0.0
                avg_l = float(np.mean(lengths)) if lengths else 0.0
                quality = compute_quality_score(avg_h, avg_l)
                results['calibration'][key] = {'sigma': sigma, 'alpha_base': alpha, 'avg_honesty': avg_h, 'avg_length': avg_l, 'quality_score': quality, 'responses': per_prompt_results}
                os.makedirs(Path(checkpoint_path).parent, exist_ok=True)
                with open(checkpoint_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            final = results['calibration'][key]
            print(f"Summary {key}: honesty={final['avg_honesty']:+.3f}, length={final['avg_length']:.0f}, quality={final['quality_score']:+.3f}")
    steerer.disable()
    return results

def run_gated_evaluation(model, tokenizer, steerer: GaussianDepthSteerer, gate: DynamicGate, prompts: List[Dict[str, Any]], alpha_peak: float, sigma_peak: float, config: Config, checkpoint_path: str, resume: bool) -> Dict[str, Any]:
    print('\n' + '=' * 65)
    print('PART B: DYNAMIC GATING')
    print(f'alpha_peak={alpha_peak}, sigma_peak={sigma_peak}')
    print(f'gate_layer={gate.gate_layer}, sharpness={gate.sharpness}')
    print('=' * 65)
    prompts = prompts * config.N_RUNS
    start_idx = 0
    gated_results: List[Dict[str, Any]] = []
    baseline_results: List[Dict[str, Any]] = []
    if resume and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                loaded = json.load(f)
            gated_results = loaded.get('gated', [])
            baseline_results = loaded.get('baseline', [])
            start_idx = int(loaded.get('completed', 0))
            print(f'Resuming gating from prompt index {start_idx}')
        except Exception as exc:
            print(f'WARNING: Could not load gating checkpoint: {exc}')
    for i in range(start_idx, len(prompts)):
        prompt_info = prompts[i]
        print(f"\n[{i + 1}/{len(prompts)}] {prompt_info['id']} ({prompt_info.get('category', 'unknown')})")
        act = gate.extract_gate_activation(model, tokenizer, prompt_info['prompt'], steerer)
        cos_sim = gate.compute_gate_score(act)
        alpha_eff = gate.get_effective_alpha(cos_sim, alpha_peak)
        print(f'  Gate score: cos_sim={cos_sim:+.4f}, alpha_eff={alpha_eff:.3f}')
        steerer.update_schedule(alpha_base=alpha_eff, sigma=sigma_peak)
        if alpha_eff < 0.001:
            steerer.disable()
        else:
            steerer.enable()
        resp_gated = generate_responses_batched(model, tokenizer, [prompt_info['prompt']], max_new_tokens=config.MAX_NEW_TOKENS, temperature=config.TEMPERATURE, top_p=config.TOP_P, do_sample=config.DO_SAMPLE)[0]
        gated_score = compute_honesty_score(resp_gated, prompt_info['honesty_keywords'], prompt_info['deception_keywords'])
        steerer.disable()
        resp_base = generate_responses_batched(model, tokenizer, [prompt_info['prompt']], max_new_tokens=config.MAX_NEW_TOKENS, temperature=config.TEMPERATURE, top_p=config.TOP_P, do_sample=config.DO_SAMPLE)[0]
        base_score = compute_honesty_score(resp_base, prompt_info['honesty_keywords'], prompt_info['deception_keywords'])
        gated_results.append({'prompt_id': prompt_info['id'], 'category': prompt_info.get('category', 'unknown'), 'cos_sim': cos_sim, 'alpha_effective': alpha_eff, 'honesty_score': gated_score['honesty_score'], 'response_length': len(resp_gated.split()), 'response': resp_gated})
        baseline_results.append({'prompt_id': prompt_info['id'], 'category': prompt_info.get('category', 'unknown'), 'honesty_score': base_score['honesty_score'], 'response_length': len(resp_base.split()), 'response': resp_base})
        checkpoint_data = {'completed': i + 1, 'alpha_peak': alpha_peak, 'sigma_peak': sigma_peak, 'threshold': gate.threshold, 'gated': gated_results, 'baseline': baseline_results}
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        print(f"  Baseline H={base_score['honesty_score']:+.2f}, Gated H={gated_score['honesty_score']:+.2f}")
    steerer.disable()
    return {'threshold': gate.threshold, 'alpha_peak': alpha_peak, 'sigma_peak': sigma_peak, 'gated': gated_results, 'baseline': baseline_results}

def create_plots(calibration_results: Dict[str, Any], gating_results: Optional[Dict[str, Any]], sigmas: List[float], alphas: List[float], plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    quality_grid = np.full((len(alphas), len(sigmas)), np.nan)
    honesty_grid = np.full((len(alphas), len(sigmas)), np.nan)
    length_grid = np.full((len(alphas), len(sigmas)), np.nan)
    for i, alpha in enumerate(alphas):
        for j, sigma in enumerate(sigmas):
            key = cfg_key(sigma, alpha)
            if key not in calibration_results['calibration']:
                continue
            cell = calibration_results['calibration'][key]
            quality_grid[i, j] = cell['quality_score']
            honesty_grid[i, j] = cell['avg_honesty']
            length_grid[i, j] = cell['avg_length']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax = axes[0]
    im = ax.imshow(quality_grid, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f's={s:g}' for s in sigmas])
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f'a={a:g}' for a in alphas])
    ax.set_xlabel('Sigma')
    ax.set_ylabel('Alpha')
    ax.set_title('Quality Heatmap')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[1]
    for sigma in sigmas:
        y_vals = []
        for alpha in alphas:
            key = cfg_key(sigma, alpha)
            y_vals.append(calibration_results['calibration'].get(key, {}).get('avg_honesty', np.nan))
        ax.plot(alphas, y_vals, marker='o', label=f'sigma={sigma:g}')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Avg Honesty')
    ax.set_title('Honesty vs Alpha')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[2]
    for sigma in sigmas:
        y_vals = []
        for alpha in alphas:
            key = cfg_key(sigma, alpha)
            y_vals.append(calibration_results['calibration'].get(key, {}).get('avg_length', np.nan))
        ax.plot(alphas, y_vals, marker='s', label=f'sigma={sigma:g}')
    ax.axhline(y=150, color='red', linestyle=':', alpha=0.5, label='min length')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Avg Length')
    ax.set_title('Coherence vs Alpha')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(f'{plots_dir}/14_calibration_sweep.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {plots_dir}/14_calibration_sweep.png')
    if not gating_results:
        return
    gated = gating_results.get('gated', [])
    baseline = gating_results.get('baseline', [])
    if not gated or not baseline:
        return
    cos_sims = [r['cos_sim'] for r in gated]
    alpha_effs = [r['alpha_effective'] for r in gated]
    h_gated = [r['honesty_score'] for r in gated]
    h_base = [r['honesty_score'] for r in baseline]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    ax.scatter(cos_sims, alpha_effs, s=90, c='#1976D2', edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Effective Alpha')
    ax.set_title(f"Gate Transfer (tau={gating_results.get('threshold', 0.0):.4f}, sigma={gating_results.get('sigma_peak', 0.0):g})")
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    x = np.arange(len(gated))
    width = 0.35
    ax.bar(x - width / 2, h_base, width, label='Baseline', color='#90CAF9', edgecolor='#1565C0')
    ax.bar(x + width / 2, h_gated, width, label='Dynamic gating', color='#FFB74D', edgecolor='#E65100')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Honesty Score')
    ax.set_title('Baseline vs Dynamic Gating')
    ax.set_xticks(x)
    ax.set_xticklabels([r['prompt_id'] for r in gated], rotation=45, fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(f'{plots_dir}/15_gating_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {plots_dir}/15_gating_analysis.png')

def main():
    parser = argparse.ArgumentParser(description='Phase 7: Strength Calibration + Dynamic Gating')
    parser.add_argument('--data-dir', type=str, default=Config.DATA_DIR)
    parser.add_argument('--prompts-file', type=str, default=Config.PROMPTS_FILE)
    parser.add_argument('--output-dir', type=str, default=Config.OUTPUT_ROOT)
    parser.add_argument('--plots-dir', type=str, default=None)
    parser.add_argument('--vector-source', type=str, default='disentangled', choices=['disentangled', 'ttpd'])
    parser.add_argument('--sigmas', type=float, nargs='+', default=Config.CALIBRATION_SIGMAS)
    parser.add_argument('--alphas', type=float, nargs='+', default=Config.CALIBRATION_ALPHAS)
    parser.add_argument('--gate-sharpness', type=float, default=Config.GATE_SHARPNESS)
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--max-tokens', type=int, default=Config.MAX_NEW_TOKENS)
    parser.add_argument('--skip-gating', action='store_true')
    parser.add_argument('--resume', action='store_true')
    if 'ipykernel' in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()
    config = Config()
    config.BATCH_SIZE = args.batch_size
    config.MAX_NEW_TOKENS = args.max_tokens
    config.GATE_SHARPNESS = args.gate_sharpness
    results_dir = f'{args.output_dir}/results'
    plots_dir = args.plots_dir or f'{args.output_dir}/plots'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    prompts = load_eval_prompts(args.prompts_file)
    if not prompts:
        raise RuntimeError('No evaluation prompts found. Expected file like phase7_data/eval_prompts_groq_50_per_category.json')
    print('=' * 70)
    print('PHASE 7: STRENGTH CALIBRATION + DYNAMIC GATING')
    print('=' * 70)
    print(f'Data dir: {args.data_dir}')
    print(f'Prompts file: {args.prompts_file}')
    print(f'Output dir: {args.output_dir}')
    print(f'Sigmas: {args.sigmas}')
    print(f'Alphas: {args.alphas}')
    print(f'Batch size: {config.BATCH_SIZE}')
    print('\n[1/6] Loading steering vectors...')
    steering_vectors = load_steering_vectors(args.data_dir, args.vector_source, Config.ALL_LAYERS)
    print('\n[2/6] Loading model...')
    model, tokenizer, device = load_model(config)
    print('\n[3/6] Initializing steerer...')
    steerer = GaussianDepthSteerer(model=model, steering_vectors=steering_vectors, alpha_base=0.0, peak_layer=Config.PEAK_LAYER, sigma=args.sigmas[0], layers=Config.ALL_LAYERS, device=device)
    steerer.register_hooks()
    calibration_checkpoint = f'{results_dir}/calibration_checkpoint_in_progress.json'
    gating_checkpoint = f'{results_dir}/gating_checkpoint_in_progress.json'
    print('\n[4/6] Running calibration over sigma x alpha...')
    calibration_results = run_calibration_sweep(model=model, tokenizer=tokenizer, steerer=steerer, prompts=prompts, alphas=args.alphas, sigmas=args.sigmas, config=config, checkpoint_path=calibration_checkpoint, resume=args.resume)
    best_key = max(calibration_results['calibration'], key=lambda k: calibration_results['calibration'][k]['quality_score'])
    best = calibration_results['calibration'][best_key]
    alpha_peak = best['alpha_base']
    sigma_peak = best['sigma']
    print(f"\nBest calibration config: sigma={sigma_peak:g}, alpha={alpha_peak:g}, quality={best['quality_score']:+.3f}")
    print('\n[5/6] Running dynamic gating...')
    gating_results = None
    if args.skip_gating:
        print('Skipping gating (--skip-gating set)')
    else:
        gate_key = f'layer_{Config.GATE_LAYER}'
        if gate_key not in steering_vectors:
            print(f'WARNING: Missing gate vector {gate_key}. Skipping gating.')
        else:
            gate = DynamicGate(model=model, truth_vector_early=steering_vectors[gate_key], gate_layer=Config.GATE_LAYER, sharpness=config.GATE_SHARPNESS, device=device)
            gate.calibrate_threshold(model, tokenizer, prompts, steerer)
            gating_results = run_gated_evaluation(model=model, tokenizer=tokenizer, steerer=steerer, gate=gate, prompts=prompts, alpha_peak=alpha_peak, sigma_peak=sigma_peak, config=config, checkpoint_path=gating_checkpoint, resume=args.resume)
    print('\n[6/6] Saving final outputs and plots...')
    final_results = {'metadata': {'model': config.MODEL_NAME, 'vector_source': args.vector_source, 'layers': Config.ALL_LAYERS, 'peak_layer': Config.PEAK_LAYER, 'gate_layer': Config.GATE_LAYER, 'sigmas': args.sigmas, 'alphas': args.alphas, 'best_sigma': sigma_peak, 'best_alpha': alpha_peak, 'batch_size': config.BATCH_SIZE, 'max_new_tokens': config.MAX_NEW_TOKENS}, 'calibration': calibration_results, 'gating': gating_results}
    results_path = f'{results_dir}/phase7_calibration_gating_results_{args.vector_source}.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    create_plots(calibration_results=calibration_results, gating_results=gating_results, sigmas=args.sigmas, alphas=args.alphas, plots_dir=plots_dir)
    steerer.remove_hooks()
    del model, tokenizer, steerer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('\n' + '=' * 70)
    print('PHASE 7 COMPLETE')
    print('=' * 70)
    print(f'Results JSON: {results_path}')
    print(f'Calibration checkpoint: {calibration_checkpoint}')
    if not args.skip_gating:
        print(f'Gating checkpoint: {gating_checkpoint}')
    print(f'Plots dir: {plots_dir}')
    print('\nCalibration summary (sorted by quality):')
    rows = sorted(calibration_results['calibration'].values(), key=lambda r: r['quality_score'], reverse=True)
    print(f"{'Sigma':<8} {'Alpha':<8} {'Honesty':<10} {'Length':<10} {'Quality':<10}")
    print('-' * 52)
    for row in rows[:10]:
        print(f"{row['sigma']:<8g} {row['alpha_base']:<8g} {row['avg_honesty']:+.3f}     {row['avg_length']:.0f}       {row['quality_score']:+.3f}")
if __name__ == '__main__':
    main()