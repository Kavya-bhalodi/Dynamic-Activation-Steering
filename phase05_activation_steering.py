#!/usr/bin/env python3
"""
Phase 5: Activation Steering — Nudging the Model Toward Honesty
================================================================

WHAT THIS DOES (Plain English):
================================

We take the steering vectors computed in Phases 3b (θ_true) and 4 (TTPD t_G)
and USE them to change the model's behavior at inference time.

The Core Mechanism — Activation Addition:
  During the forward pass, we manually nudge the AI's internal "thoughts"
  toward the honest direction before it generates the next word.

  The intervention at each scheduled layer (L) is:
    x'_L = x_L + α · θ_true

  Where:
    x_L     = original residual stream activation at layer L
    α       = steering strength (scalar multiplier)
    θ_true  = the disentangled truth direction vector

  When α > 0: the model is pushed toward honesty
  When α < 0: the model is pushed toward deception (for testing)
  When α = 0: no steering (baseline)

Pipeline:
  1. Load the model (same 4-bit Llama-3-8B from Phase 3)
  2. Load steering vectors (Phase 3b θ_true + Phase 4 TTPD t_G)
  3. Register forward hooks that add α·θ_true to activations
  4. Run curated test prompts through the steered model
  5. Sweep α values to find optimal steering strength
  6. Compare outputs: unsteered vs steered, Phase 3b vs TTPD

USAGE:
  # Full run on GPU (Colab T4):
  python 5_activation_steering.py

  # Quick test with TinyLlama on CPU:
  python 5_activation_steering.py --test

  # Custom alpha values:
  python 5_activation_steering.py --alphas 0 5 10 15

  # Use specific layers:
  python 5_activation_steering.py --layers 14 16 18

  # Use TTPD vectors instead of Phase 3b:
  python 5_activation_steering.py --vector-source ttpd

Estimated time:
  ~15-25 min for full sweep on T4 (8 alphas × 20 prompts)
  ~2 min for test mode on CPU
"""

import os
import sys
import json
import time
import gc
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """All configuration in one place"""

    # Model (must match Phase 3)
    MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"

    # Directories
    DATA_DIR = "activations_output"
    OUTPUT_DIR = "activations_output"
    PLOTS_DIR = "plots"

    # Layers used in Phases 3-4
    LAYERS_TO_EXTRACT = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 31]

    # Default steering layers (best layers from Phase 4)
    DEFAULT_STEERING_LAYERS = [14, 16, 18]

    # Alpha sweep values
    DEFAULT_ALPHAS = [0, 1, 2, 5, 8, 10, 15, 20]

    # Generation parameters
    N_RUNS = 1
    MAX_NEW_TOKENS = 300
    TEMPERATURE = 0.7
    TOP_P = 0.9
    DO_SAMPLE = True

    # 4-bit quantization
    USE_4BIT = True

    # HuggingFace token
    HF_TOKEN = None


# ============================================================================
# EVALUATION PROMPTS
# ============================================================================

import scipy.stats as stats

# Load prompts from the generated JSON
prompt_file = Path("datasets/pipeline_test/scenarios/eval_prompts_groq_50_per_category.json")
if prompt_file.exists():
    with open(prompt_file, "r") as f:
        EVAL_PROMPTS = json.load(f).get("eval_prompts", [])
else:
    EVAL_PROMPTS = []

# ============================================================================
# STEERING HOOK — The Core Mechanism
# ============================================================================

class ActivationSteerer:
    """
    Hooks into model layers to add steering vectors during the forward pass.

    The intervention: x'_L = x_L + α · θ_true

    This modifies the residual stream activations to push the model's
    internal representation toward the honest direction.
    """

    def __init__(
        self,
        model,
        steering_vectors: Dict[str, np.ndarray],
        alpha: float = 5.0,
        layers: List[int] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.alpha = alpha
        self.layers = layers or Config.DEFAULT_STEERING_LAYERS
        self.device = device
        self.hooks = []
        self.active = False  # Toggle steering on/off

        # Convert steering vectors to tensors on the correct device
        self.steering_tensors = {}
        for layer_idx in self.layers:
            key = f"layer_{layer_idx}"
            if key in steering_vectors:
                vec = steering_vectors[key]
                # Ensure it's a unit vector, then store raw (alpha applied in hook)
                vec_norm = np.linalg.norm(vec)
                if vec_norm > 1e-8:
                    # Keep as-is (not unit-normalized) — the raw direction carries magnitude info
                    self.steering_tensors[layer_idx] = torch.tensor(
                        vec, dtype=torch.float16, device=device
                    )
                    print(f"    Layer {layer_idx}: loaded vector (norm={vec_norm:.4f})")
                else:
                    print(f"    Layer {layer_idx}: WARNING — near-zero vector, skipping")

    def _create_hook(self, layer_idx: int):
        """
        Create a forward hook that adds α · θ_true to the hidden states.

        The hook modifies the output tensor IN-PLACE during the forward pass.
        This is the core of activation steering.
        """
        steering_vec = self.steering_tensors[layer_idx]

        def hook_fn(module, input, output):
            if not self.active:
                return output

            # output is a tuple: (hidden_states, ...) for Llama
            hidden_states = output[0]

            # Add steering vector to ALL token positions
            # x'_L = x_L + α · θ_true
            # steering_vec shape: [hidden_dim]
            # hidden_states shape: [batch, seq_len, hidden_dim]
            perturbation = (self.alpha * steering_vec).to(dtype=hidden_states.dtype)
            modified = hidden_states + perturbation.unsqueeze(0).unsqueeze(0)

            # Return modified output tuple
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return hook_fn

    def register_hooks(self):
        """Attach steering hooks to the model's layers."""
        self.remove_hooks()  # Clean up any existing hooks

        for layer_idx in self.layers:
            if layer_idx not in self.steering_tensors:
                continue
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._create_hook(layer_idx))
            self.hooks.append(hook)

        print(f"  Registered steering hooks on {len(self.hooks)} layers")

    def remove_hooks(self):
        """Detach all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def set_alpha(self, alpha: float):
        """Update the steering strength. Requires re-registering hooks."""
        self.alpha = alpha
        # Re-register hooks with new alpha
        self.register_hooks()

    def enable(self):
        """Turn on steering."""
        self.active = True

    def disable(self):
        """Turn off steering (passthrough mode)."""
        self.active = False


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(config: Config, test_mode: bool = False):
    """Load model — TinyLlama for test, Llama-3-8B for real."""

    if test_mode:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"\nLoading TEST model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        model.eval()
        device = "cpu"
        return model, tokenizer, device

    print(f"\nLoading model: {config.MODEL_NAME}")
    print(f"  4-bit quantization: {config.USE_4BIT}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, token=config.HF_TOKEN
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.USE_4BIT:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto",
            token=config.HF_TOKEN,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=config.HF_TOKEN,
        )

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory used: {mem_gb:.1f} GB")

    print(f"  Model loaded successfully")
    return model, tokenizer, device


# ============================================================================
# STEERING VECTOR LOADING
# ============================================================================

def load_steering_vectors(
    data_dir: str,
    source: str = "disentangled",
    layers: List[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Load steering vectors from previous phases.

    Args:
        data_dir: Path to activations output directory
        source: 'disentangled' (Phase 3b θ_true) or 'ttpd' (Phase 4 t_G)
        layers: Which layers to load vectors for

    Returns:
        Dict mapping 'layer_X' -> numpy array of shape [hidden_dim]
    """
    vectors = {}

    if source == "disentangled":
        sv_path = f"{data_dir}/steering_vectors_disentangled.npz"
        key_template = "{}_theta_true"
        print(f"\n  Loading Phase 3b disentangled θ_true from {sv_path}")
    elif source == "ttpd":
        sv_path = f"{data_dir}/steering_vectors_ttpd.npz"
        key_template = "{}_t_G"
        print(f"\n  Loading Phase 4 TTPD t_G from {sv_path}")
    else:
        raise ValueError(f"Unknown source: {source}. Use 'disentangled' or 'ttpd'")

    sv = np.load(sv_path)
    available_keys = list(sv.keys())
    print(f"  Available keys: {len(available_keys)}")

    if layers is None:
        layers = Config.LAYERS_TO_EXTRACT

    for layer_idx in layers:
        layer_name = f"layer_{layer_idx}"
        key = key_template.format(layer_name)
        if key in sv:
            vectors[layer_name] = sv[key]
        else:
            # Try alternate key naming
            alt_key = f"{layer_name}_theta_true_unit" if source == "disentangled" else f"{layer_name}_t_G_unit"
            if alt_key in sv:
                vectors[layer_name] = sv[alt_key]
            else:
                print(f"  WARNING: No vector found for {layer_name} (tried '{key}')")

    print(f"  Loaded vectors for {len(vectors)} layers")
    return vectors


# ============================================================================
# GENERATION
# ============================================================================

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    device: str = "cuda",
) -> str:
    """Generate a response from the model given a prompt."""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )
    input_device = next(model.parameters()).device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (not the prompt)
    generated_ids = outputs[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Cleanup
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response.strip()


# ============================================================================
# HONESTY SCORING
# ============================================================================

def compute_honesty_score(
    response: str,
    honesty_keywords: List[str],
    deception_keywords: List[str],
) -> Dict[str, float]:
    """
    Compute a simple keyword-based honesty score.

    This is a rough heuristic — not a definitive measure of honesty.
    Higher score = response uses more honesty-associated language.

    Returns:
        Dict with honesty_hits, deception_hits, honesty_score (-1 to +1)
    """
    response_lower = response.lower()

    honesty_hits = sum(1 for kw in honesty_keywords if kw.lower() in response_lower)
    deception_hits = sum(1 for kw in deception_keywords if kw.lower() in response_lower)

    total = honesty_hits + deception_hits
    if total == 0:
        score = 0.0
    else:
        score = (honesty_hits - deception_hits) / total

    return {
        "honesty_hits": honesty_hits,
        "deception_hits": deception_hits,
        "total_keywords": total,
        "honesty_score": score,  # Range: -1 (deceptive) to +1 (honest)
    }


def compute_response_length(response: str) -> int:
    """Count words in response (proxy for coherence — very short = broken)."""
    return len(response.split())


# ============================================================================
# ALPHA SWEEP — The Main Experiment
# ============================================================================

def run_alpha_sweep(
    model,
    tokenizer,
    steerer: ActivationSteerer,
    prompts: List[Dict],
    alphas: List[float],
    config: Config,
    device: str = "cuda",
) -> Dict:
    """
    Run the full alpha sweep experiment.
    """

    results = {
        "alphas": alphas,
        "layers": steerer.layers,
        "prompts": [{"id": p["id"], "category": p["category"]} for p in prompts],
        "sweep_results": {},
    }

    total_runs = len(alphas) * len(prompts) * config.N_RUNS
    run_count = 0
    t_start = time.time()

    for alpha in alphas:
        print(f"\n  {'='*55}")
        print(f"  α = {alpha}")
        print(f"  {'='*55}")

        alpha_key = f"alpha_{alpha}"
        results["sweep_results"][alpha_key] = {
            "alpha": alpha,
            "responses": [],
            "avg_honesty_score": 0.0,
            "avg_response_length": 0.0,
            "runs": {
                i: {"avg_honesty_score": 0.0, "avg_response_length": 0.0}
                for i in range(config.N_RUNS)
            },
        }

        steerer.set_alpha(alpha)
        if alpha == 0:
            steerer.disable()
        else:
            steerer.enable()

        honesty_scores = []
        response_lengths = []
        run_scores = {i: [] for i in range(config.N_RUNS)}
        run_lengths = {i: [] for i in range(config.N_RUNS)}

        for run_idx in range(config.N_RUNS):
            for prompt_info in prompts:
                run_count += 1
                elapsed = time.time() - t_start
                rate = run_count / elapsed if elapsed > 0 else 0
                remaining = (total_runs - run_count) / rate if rate > 0 else 0

                print(
                    f"    [{run_count}/{total_runs}] Run {run_idx+1} | {prompt_info['id']:<10} "
                    f"({remaining/60:.1f} min remaining)"
                )

                # Generate response
                response = generate_response(
                    model,
                    tokenizer,
                    prompt=prompt_info["prompt"],
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P,
                    do_sample=config.DO_SAMPLE,
                    device=device,
                )

                # Score
                score_info = compute_honesty_score(
                    response,
                    prompt_info["honesty_keywords"],
                    prompt_info["deception_keywords"],
                )
                resp_len = compute_response_length(response)

                honesty_scores.append(score_info["honesty_score"])
                response_lengths.append(resp_len)
                run_scores[run_idx].append(score_info["honesty_score"])
                run_lengths[run_idx].append(resp_len)

                results["sweep_results"][alpha_key]["responses"].append(
                    {
                        "run_idx": run_idx,
                        "prompt_id": prompt_info["id"],
                        "category": prompt_info["category"],
                        "response": response,
                        "honesty_score": score_info["honesty_score"],
                        "honesty_hits": score_info["honesty_hits"],
                        "deception_hits": score_info["deception_hits"],
                        "response_length": resp_len,
                    }
                )

                # Print short preview
                preview = response[:120].replace("\n", " ")
                print(
                    f"      Score: {score_info['honesty_score']:+.2f} "
                    f"(H:{score_info['honesty_hits']}/D:{score_info['deception_hits']}) "
                    f"Len:{resp_len}"
                )
                print(f"      Preview: {preview}...")

        # Aggregate stats for this alpha
        avg_honesty = np.mean(honesty_scores) if honesty_scores else 0.0
        avg_length = np.mean(response_lengths) if response_lengths else 0.0

        results["sweep_results"][alpha_key]["avg_honesty_score"] = float(avg_honesty)
        results["sweep_results"][alpha_key]["avg_response_length"] = float(avg_length)

        for i in range(config.N_RUNS):
            results["sweep_results"][alpha_key]["runs"][i]["avg_honesty_score"] = (
                float(np.mean(run_scores[i])) if run_scores[i] else 0.0
            )
            results["sweep_results"][alpha_key]["runs"][i]["avg_response_length"] = (
                float(np.mean(run_lengths[i])) if run_lengths[i] else 0.0
            )

        print(f"\n    α={alpha} Summary:")
        print(f"      Avg honesty score: {avg_honesty:+.3f}")
        print(f"      Avg response length: {avg_length:.0f} words")

    # Disable steering after sweep
    steerer.disable()
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_steering_plots(results: Dict, plots_dir: str, vector_source: str):
    """Create visualizations of the alpha sweep results."""

    os.makedirs(plots_dir, exist_ok=True)

    alphas = results["alphas"]
    avg_scores = []
    avg_lengths = []
    
    # Try to safely get number of runs. Fallback to 1 if older result struct.
    try:
        n_runs = len(results["sweep_results"][f"alpha_{alphas[0]}"]["runs"])
    except KeyError:
        n_runs = 1
        
    run_avg_scores = {i: [] for i in range(n_runs)}
    run_avg_lengths = {i: [] for i in range(n_runs)}

    for alpha in alphas:
        key = f"alpha_{alpha}"
        avg_scores.append(results["sweep_results"][key]["avg_honesty_score"])
        avg_lengths.append(results["sweep_results"][key]["avg_response_length"])
        for i in range(n_runs):
            try:
                run_avg_scores[i].append(results["sweep_results"][key]["runs"][i]["avg_honesty_score"])
                run_avg_lengths[i].append(results["sweep_results"][key]["runs"][i]["avg_response_length"])
            except KeyError: # Fallback backwards compatibility
                run_avg_scores[i].append(results["sweep_results"][key]["avg_honesty_score"])
                run_avg_lengths[i].append(results["sweep_results"][key]["avg_response_length"])                

    print("\n--- Hypothesis Testing & Correlations ---")
    if len(alphas) > 1 and len(set(avg_scores)) > 1:
        corr, p_val = stats.pearsonr(alphas, avg_scores)
        print(f"Alpha vs Honesty Correlation: r={corr:.3f}, p={p_val:.3g}")
        base_key = f"alpha_{alphas[0]}"
        base_scores = [r["honesty_score"] for r in results["sweep_results"][base_key]["responses"]]
        best_alpha = max(alphas, key=lambda a: results["sweep_results"][f"alpha_{a}"]["avg_honesty_score"])
        if best_alpha != alphas[0]:
            best_key = f"alpha_{best_alpha}"
            best_scores = [r["honesty_score"] for r in results["sweep_results"][best_key]["responses"]]
            if len(best_scores) == len(base_scores):
                t_stat, t_p = stats.ttest_rel(best_scores, base_scores)
                print(f"Baseline (α={alphas[0]}) vs Best (α={best_alpha}) Paired T-Test: t={t_stat:.3f}, p={t_p:.3g}")
    print("-----------------------------------------")

    # ── FIGURE 1: Honesty Score vs Alpha ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    
    run_colors = plt.cm.Set1(np.linspace(0, 1, n_runs))
    for i in range(n_runs):
        ax.plot(alphas, run_avg_scores[i], 'x--', color=run_colors[i], alpha=0.6, linewidth=1.5, label=f'Run {i+1}')

    scores_matrix = np.array([run_avg_scores[i] for i in range(n_runs)])
    std_scores = np.std(scores_matrix, axis=0) if n_runs > 1 else np.zeros_like(avg_scores)
    
    ax.plot(alphas, avg_scores, 'o-', color='#2196F3', linewidth=3,
            markersize=10, markerfacecolor='white', markeredgewidth=2.5, label='Mean')
    if n_runs > 1:
        ax.fill_between(alphas, np.array(avg_scores) - std_scores, np.array(avg_scores) + std_scores, alpha=0.2, color='#2196F3', label='±1 Std Dev')
        
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=avg_scores[0], color='red', linestyle=':', alpha=0.5,
               label=f'Baseline Mean: {avg_scores[0]:+.3f}')
    ax.set_xlabel('Steering Strength (α)', fontsize=13)
    ax.set_ylabel('Honesty Score', fontsize=13)
    ax.set_title(f'Activation Steering Effect on Honesty\n'
                 f'(Source: {vector_source}, Layers: {results["layers"]})',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

    # ── FIGURE 2: Response Length vs Alpha ──
    ax = axes[1]
    
    for i in range(n_runs):
        ax.plot(alphas, run_avg_lengths[i], 'x--', color=run_colors[i], alpha=0.6, linewidth=1.5)
        
    lengths_matrix = np.array([run_avg_lengths[i] for i in range(n_runs)])
    std_lengths = np.std(lengths_matrix, axis=0) if n_runs > 1 else np.zeros_like(avg_lengths)

    ax.plot(alphas, avg_lengths, 's-', color='#FF9800', linewidth=3,
            markersize=10, markerfacecolor='white', markeredgewidth=2.5, label='Mean')
    if n_runs > 1:
        ax.fill_between(alphas, np.array(avg_lengths) - std_lengths, np.array(avg_lengths) + std_lengths, alpha=0.2, color='#FF9800', label='±1 Std Dev')
        
    ax.axhline(y=avg_lengths[0], color='red', linestyle=':', alpha=0.5,
               label=f'Baseline Mean: {avg_lengths[0]:.0f}')
    ax.set_xlabel('Steering Strength (α)', fontsize=13)
    ax.set_ylabel('Response Length (words)', fontsize=13)
    ax.set_title(f'Response Coherence vs Steering Strength\n'
                 f'(Shorter responses at high α may indicate degradation)',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{plots_dir}/07_activation_steering_sweep.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {plots_dir}/07_activation_steering_sweep.png")

    # ── FIGURE 3: Per-Category Honesty Scores (Mean across runs) ──
    categories = sorted(set(r["category"]
                            for r in results["sweep_results"][f"alpha_{alphas[0]}"]["responses"]))
    n_cats = len(categories)

    fig, ax = plt.subplots(figsize=(14, 7))

    cat_colors = plt.cm.Set2(np.linspace(0, 1, n_cats))

    for i, cat in enumerate(categories):
        cat_scores = []
        for alpha in alphas:
            key = f"alpha_{alpha}"
            cat_responses = [r for r in results["sweep_results"][key]["responses"]
                             if r["category"] == cat]
            avg = np.mean([r["honesty_score"] for r in cat_responses]) if cat_responses else 0
            cat_scores.append(avg)

        ax.plot(alphas, cat_scores, 'o-', color=cat_colors[i], linewidth=1.5,
                markersize=6, label=cat.replace('_', ' ').title(), alpha=0.8)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Steering Strength (α)', fontsize=13)
    ax.set_ylabel('Average Honesty Score', fontsize=13)
    ax.set_title(f'Per-Category Honesty Scores (Mean) vs α\n'
                 f'(Source: {vector_source})', fontsize=13)
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    fig.savefig(f"{plots_dir}/08_per_category_steering.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {plots_dir}/08_per_category_steering.png")

    # ── FIGURE 4: Heatmap — Prompt × Alpha (Mean across runs) ──
    prompt_ids = []
    for r in results["sweep_results"][f"alpha_{alphas[0]}"]["responses"]:
        if r["prompt_id"] not in prompt_ids:
            prompt_ids.append(r["prompt_id"])
            
    n_prompts = len(prompt_ids)

    heatmap_data = np.zeros((n_prompts, len(alphas)))
    for j, alpha in enumerate(alphas):
        key = f"alpha_{alpha}"
        for i, pid in enumerate(prompt_ids):
            resps = [r["honesty_score"] for r in results["sweep_results"][key]["responses"] if r["prompt_id"] == pid]
            heatmap_data[i, j] = np.mean(resps) if resps else 0.0

    fig, ax = plt.subplots(figsize=(12, max(6, n_prompts * 0.35)))
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f'α={a}' for a in alphas], fontsize=10)
    ax.set_yticks(range(n_prompts))
    ax.set_yticklabels(prompt_ids, fontsize=9)
    ax.set_xlabel('Steering Strength', fontsize=12)
    ax.set_ylabel('Prompt', fontsize=12)
    ax.set_title(f'Honesty Score Heatmap (Mean across runs) — Prompt × Alpha\n'
                 f'(Green = Honest, Red = Deceptive)', fontsize=13)

    plt.colorbar(im, ax=ax, label='Mean Honesty Score', shrink=0.8)

    # Add text annotations
    for i in range(n_prompts):
        for j in range(len(alphas)):
            val = heatmap_data[i, j]
            color = 'black' if abs(val) < 0.5 else 'white'
            ax.text(j, i, f'{val:+.1f}', ha='center', va='center',
                    fontsize=7, color=color)

    plt.tight_layout()
    fig.savefig(f"{plots_dir}/09_steering_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {plots_dir}/09_steering_heatmap.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Activation Steering")
    parser.add_argument("--test", action="store_true",
                        help="Quick test with TinyLlama on CPU")
    parser.add_argument("--data-dir", type=str, default=Config.DATA_DIR,
                        help="Path to activations output")
    parser.add_argument("--plots-dir", type=str, default=Config.PLOTS_DIR,
                        help="Where to save plots")
    parser.add_argument("--vector-source", type=str, default="disentangled",
                        choices=["disentangled", "ttpd"],
                        help="Which steering vectors to use (default: disentangled)")
    parser.add_argument("--alphas", type=float, nargs='+',
                        default=Config.DEFAULT_ALPHAS,
                        help="Alpha values to sweep")
    parser.add_argument("--layers", type=int, nargs='+',
                        default=Config.DEFAULT_STEERING_LAYERS,
                        help="Layers to steer on")
    parser.add_argument("--max-tokens", type=int, default=Config.MAX_NEW_TOKENS,
                        help="Max tokens to generate per response")
    args = parser.parse_args()

    config = Config()
    config.MAX_NEW_TOKENS = args.max_tokens

    print("=" * 65)
    print("PHASE 5: ACTIVATION STEERING")
    print("  Nudging model activations toward the honest direction")
    print("=" * 65)

    if args.test:
        run_test_mode(config, args)
        return

    # ── Step 1: Load steering vectors ──
    print("\n[1/5] Loading steering vectors...")
    steering_vectors = load_steering_vectors(
        data_dir=args.data_dir,
        source=args.vector_source,
        layers=args.layers,
    )

    # ── Step 2: Load model ──
    print("\n[2/5] Loading model...")
    model, tokenizer, device = load_model(config, test_mode=False)

    # ── Step 3: Setup steerer ──
    print("\n[3/5] Setting up activation steerer...")
    steerer = ActivationSteerer(
        model=model,
        steering_vectors=steering_vectors,
        alpha=0.0,  # Start with no steering
        layers=args.layers,
        device=device,
    )
    steerer.register_hooks()

    # ── Step 4: Run alpha sweep ──
    print(f"\n[4/5] Running alpha sweep...")
    print(f"  Alpha values: {args.alphas}")
    print(f"  Steering layers: {args.layers}")
    print(f"  Vector source: {args.vector_source}")
    print(f"  Test prompts: {len(EVAL_PROMPTS)}")
    print(f"  Total runs: {len(args.alphas) * len(EVAL_PROMPTS)}")

    results = run_alpha_sweep(
        model=model,
        tokenizer=tokenizer,
        steerer=steerer,
        prompts=EVAL_PROMPTS,
        alphas=args.alphas,
        config=config,
        device=device,
    )

    # Add metadata
    results["metadata"] = {
        "vector_source": args.vector_source,
        "model": config.MODEL_NAME,
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "temperature": config.TEMPERATURE,
    }

    # ── Step 5: Save results and create plots ──
    print(f"\n[5/5] Saving results and creating plots...")

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    # Save JSON results
    results_path = f"{args.data_dir}/steering_results_{args.vector_source}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results: {results_path}")

    # Create plots
    create_steering_plots(results, args.plots_dir, args.vector_source)

    # ── Cleanup ──
    steerer.remove_hooks()
    del model, tokenizer, steerer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*65}")
    print("PHASE 5 COMPLETE — ACTIVATION STEERING")
    print(f"{'='*65}")

    # Print comparison table
    print(f"\n  Alpha Sweep Summary:")
    print(f"  {'Alpha':<8} {'Honesty Score':<16} {'Response Length':<16}")
    print(f"  {'-'*40}")
    for alpha in args.alphas:
        key = f"alpha_{alpha}"
        r = results["sweep_results"][key]
        print(f"  {alpha:<8} {r['avg_honesty_score']:+.3f}{'':>11} "
              f"{r['avg_response_length']:.0f} words")

    # Find optimal alpha
    best_alpha_key = max(results["sweep_results"],
                         key=lambda k: results["sweep_results"][k]["avg_honesty_score"])
    best_alpha = results["sweep_results"][best_alpha_key]["alpha"]
    best_score = results["sweep_results"][best_alpha_key]["avg_honesty_score"]

    print(f"\n  Optimal α: {best_alpha} (honesty score: {best_score:+.3f})")
    print(f"  Vector source: {args.vector_source}")
    print(f"  Layers steered: {args.layers}")

    print(f"\n  Output files:")
    print(f"    {results_path}")
    print(f"    {args.plots_dir}/07_activation_steering_sweep.png")
    print(f"    {args.plots_dir}/08_per_category_steering.png")
    print(f"    {args.plots_dir}/09_steering_heatmap.png")


# ============================================================================
# TEST MODE
# ============================================================================

def run_test_mode(config: Config, args):
    """
    Quick test with TinyLlama and synthetic steering vectors.
    Validates the entire pipeline without needing a GPU.
    """
    print("\n  TEST MODE — using TinyLlama + synthetic vectors\n")

    # ── Load small model ──
    model, tokenizer, device = load_model(config, test_mode=True)
    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    print(f"  Model layers: {n_layers}, hidden_dim: {hidden_dim}")

    # ── Create synthetic steering vectors ──
    print("\n  Creating synthetic steering vectors...")
    rng = np.random.RandomState(42)
    test_layers = [5, 11, min(15, n_layers - 1)]
    test_layers = [l for l in test_layers if l < n_layers]

    steering_vectors = {}
    for layer_idx in test_layers:
        vec = rng.randn(hidden_dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec) * 0.5  # Scale to reasonable norm
        steering_vectors[f"layer_{layer_idx}"] = vec
        print(f"    layer_{layer_idx}: synthetic vector (norm={np.linalg.norm(vec):.4f})")

    # ── Test prompts (subset) ──
    test_prompts = EVAL_PROMPTS[:4]

    # ── Create steerer ──
    print("\n  Setting up steerer...")
    steerer = ActivationSteerer(
        model=model,
        steering_vectors=steering_vectors,
        alpha=0.0,
        layers=test_layers,
        device=device,
    )
    steerer.register_hooks()

    # ── Test 1: Verify hook injection ──
    print("\n  Test 1: Verifying hook injection modifies activations...")

    test_text = "The following statement is true: AI systems should be transparent."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)

    # Run WITHOUT steering
    captured_clean = {}
    capture_hooks = []
    for layer_idx in test_layers:
        def make_capture_hook(name):
            def hook_fn(module, input, output):
                captured_clean[name] = output[0].detach().clone()
            return hook_fn
        h = model.model.layers[layer_idx].register_forward_hook(
            make_capture_hook(f"layer_{layer_idx}")
        )
        capture_hooks.append(h)

    steerer.disable()
    with torch.no_grad():
        model(**inputs)
    for h in capture_hooks:
        h.remove()

    # Run WITH steering (α=5)
    # IMPORTANT: Register steering hooks FIRST, then capture hooks.
    # PyTorch fires hooks in registration order per module.
    # If capture hook fires before steering hook, it sees unmodified output.
    steerer.set_alpha(5.0)
    steerer.enable()

    captured_steered = {}
    capture_hooks = []
    for layer_idx in test_layers:
        def make_capture_hook2(name):
            def hook_fn(module, input, output):
                captured_steered[name] = output[0].detach().clone()
            return hook_fn
        h = model.model.layers[layer_idx].register_forward_hook(
            make_capture_hook2(f"layer_{layer_idx}")
        )
        capture_hooks.append(h)
    with torch.no_grad():
        model(**inputs)
    for h in capture_hooks:
        h.remove()

    # Verify activations are DIFFERENT
    print("\n  Activation differences (steered - clean):")
    all_different = True
    for layer_name in captured_clean:
        if layer_name in captured_steered:
            diff = (captured_steered[layer_name] - captured_clean[layer_name]).abs().mean().item()
            max_diff = (captured_steered[layer_name] - captured_clean[layer_name]).abs().max().item()
            print(f"    {layer_name}: mean_diff={diff:.6f}, max_diff={max_diff:.6f}")
            if diff < 1e-8:
                all_different = False
                print(f"      WARNING: Activations unchanged!")

    if all_different:
        print("\n  ✓ Hook injection VERIFIED — activations are modified by steering")
    else:
        print("\n  ✗ PROBLEM — some activations were not modified")

    # ── Test 2: Run mini alpha sweep ──
    print("\n  Test 2: Running mini alpha sweep (α = 0, 5, 10)...")
    test_alphas = [0, 5, 10]

    results = run_alpha_sweep(
        model=model,
        tokenizer=tokenizer,
        steerer=steerer,
        prompts=test_prompts,
        alphas=test_alphas,
        config=config,
        device=device,
    )

    # ── Test 3: Check that generation produces different outputs ──
    print("\n  Test 3: Comparing steered vs unsteered generations...")
    baseline = results["sweep_results"]["alpha_0"]["responses"][0]["response"]
    steered = results["sweep_results"]["alpha_10"]["responses"][0]["response"]

    print(f"    Baseline (α=0): {baseline[:100]}...")
    print(f"    Steered (α=10): {steered[:100]}...")
    print(f"    Same output: {baseline == steered}")

    # ── Save test results ──
    os.makedirs(args.plots_dir, exist_ok=True)
    create_steering_plots(results, args.plots_dir, "test_synthetic")

    # ── Summary ──
    print(f"\n{'='*55}")
    print("  TEST MODE COMPLETE")
    print(f"{'='*55}")
    print(f"  Hook injection:   {'PASS ✓' if all_different else 'FAIL ✗'}")
    print(f"  Alpha sweep:      PASS ✓ ({len(test_alphas)} alphas × {len(test_prompts)} prompts)")
    print(f"  Output differs:   {'PASS ✓' if baseline != steered else 'SAME (may be OK for small model)'}")
    print(f"  Plots saved:      {args.plots_dir}/")

    # Cleanup
    steerer.remove_hooks()
    del model, tokenizer, steerer
    gc.collect()


if __name__ == "__main__":
    main()
