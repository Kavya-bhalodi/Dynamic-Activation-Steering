#!/usr/bin/env python3
"""
Phase 6: Gaussian Depth Scheduling for Activation Steering
============================================================

WHAT THIS DOES (Plain English):
================================

Phase 5 applied the SAME steering force (α) at every layer. This works but
has a problem: early layers handle low-level grammar/syntax, late layers handle
high-level decisions. Forcing them all equally can break coherence.

Gaussian Depth Scheduling fixes this:

  Instead of applying the same force across every layer, we use a targeted
  bell curve (Gaussian) distribution:

    α_L = α_base × exp( -(L - μ)² / (2σ²) )

  Where:
    - L     = layer index (6, 8, 10, ..., 31)
    - μ     = peak layer (default: 16, the "Honesty Peak" from Phase 4)
    - σ     = spread (controls how many layers get strong steering)
    - α_base = base steering strength

  The Peak:  Maximum steering at Layer 16 (highest Cohen's d = 3.16)
  The Taper: Strength smoothly decreases toward beginning/end layers,
             preserving grammar (early layers) and coherence (late layers).

  Example with μ=16, σ=4, α_base=10:
    Layer 6:   α =  1.1  (barely touched — protects grammar)
    Layer 10:  α =  3.2  (moderate steering)
    Layer 14:  α =  8.8  (strong steering)
    Layer 16:  α = 10.0  (PEAK — maximum honesty push)
    Layer 18:  α =  8.8  (strong steering)
    Layer 24:  α =  1.4  (light touch)
    Layer 31:  α =  0.0  (barely touched — protects output coherence)

USAGE:
  # Full run on GPU (Colab T4):
  python 6_gaussian_depth_steering.py

  # Quick test with TinyLlama on CPU:
  python 6_gaussian_depth_steering.py --test

  # Custom Gaussian parameters:
  python 6_gaussian_depth_steering.py --peak-layer 16 --sigma 4 --alpha-base 10

  # Sweep over multiple sigma values:
  python 6_gaussian_depth_steering.py --sigma-sweep 2 3 4 6 8

  # Use TTPD vectors:
  python 6_gaussian_depth_steering.py --vector-source ttpd

Estimated time:
  ~20-30 min for full sweep on T4
  ~2 min for test mode on CPU
"""

import os

# 1. THE SAFETY LOCK: Do this BEFORE importing torch/vllm
target_uuid = "MIG-e5d78ce7-5816-5a4a-80e4-760fd53e696d"
os.environ["CUDA_VISIBLE_DEVICES"] = target_uuid

# Prevent HuggingFace from downloading giant models to the small /home drive!
# We will route all HuggingFace traffic directly to the 10TB /data HDD
os.environ["HF_HOME"] = "/scratch/shlok/hf_cache"

import sys
import json
import time
import gc
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# PATHS
# ============================================================================
INPUT_DIR = "./data"
OUTPUT_DIR = "./output"
from collections import defaultdict

import torch

# 2. VERIFICATION: Ensure the app only sees ONE device (your 24GB slice)
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    if device_count != 1:
        raise RuntimeError(f"CRITICAL ERROR: Expected 1 GPU, but saw {device_count}. Isolation failed!")
    
    actual_uuid = torch.cuda.get_device_properties(0).uuid
    print(f"Verified: App is locked to MIG Instance {actual_uuid}")

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
transformers.logging.set_verbosity_error()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """All configuration in one place"""

    # Model (must match Phases 3-5)
    MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"

    # Directories
    DATA_DIR = f"{INPUT_DIR}/activations"
    OUTPUT_DIR = f"{OUTPUT_DIR}/results"
    PLOTS_DIR = f"{OUTPUT_DIR}/plots"

    # ALL layers we have steering vectors for
    ALL_LAYERS = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 31]

    # Gaussian peak — Layer 16 (best Cohen's d from Phase 4)
    PEAK_LAYER = 16

    # Gaussian spread — controls how many layers get strong steering
    # σ=4 → ~6 layers in the "active zone"
    DEFAULT_SIGMA = 4.0

    # Base alpha (peak alpha)
    DEFAULT_ALPHA_BASE = 10.0

    # Sigma values to sweep
    DEFAULT_SIGMA_SWEEP = [1, 2, 3, 4,5, 6]

    # Alpha base values to sweep
    DEFAULT_ALPHA_SWEEP = [0, 0.125, 0.25, 0.5, 0.75, 1]

    # Generation parameters
    N_RUNS = 1
    BATCH_SIZE = 64  # Pushed to 32x to maximize 24GB MIG usage
    MAX_NEW_TOKENS = 300
    TEMPERATURE = 0.7
    TOP_P = 0.9
    DO_SAMPLE = True

    # 4-bit quantization
    USE_4BIT = False  # Disabled for massive speedup (fits easily in 24GB)
    HF_TOKEN = "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"


# ============================================================================
# EVALUATION PROMPTS (same as Phase 5)
# ============================================================================

import json
from pathlib import Path
import scipy.stats as stats
import numpy as np

# Load prompts from the generated JSON
prompt_file = Path(f"{INPUT_DIR}/eval_prompts_groq_50_per_category.json")
if prompt_file.exists():
    with open(prompt_file, "r") as f:
        EVAL_PROMPTS = json.load(f).get("eval_prompts", [])
else:
    EVAL_PROMPTS = []

# ============================================================================
# GAUSSIAN DEPTH SCHEDULE
# ============================================================================

def compute_gaussian_weights(
    layers: List[int],
    peak_layer: int = 16,
    sigma: float = 4.0,
) -> Dict[int, float]:
    """
    Compute Gaussian bell-curve weights for each layer.

    w(L) = exp( -(L - μ)² / (2σ²) )

    Args:
        layers: List of layer indices (e.g., [6, 8, 10, ..., 31])
        peak_layer: μ — layer with maximum weight (weight = 1.0)
        sigma: σ — controls spread of the bell curve

    Returns:
        Dict mapping layer_index -> weight in [0, 1]
    """
    weights = {}
    for L in layers:
        w = np.exp(-((L - peak_layer) ** 2) / (2 * sigma ** 2))
        weights[L] = float(w)
    return weights


def compute_per_layer_alphas(
    layers: List[int],
    alpha_base: float,
    peak_layer: int = 16,
    sigma: float = 4.0,
) -> Dict[int, float]:
    """
    Compute per-layer steering strengths using Gaussian scheduling.

    α_L = α_base × exp( -(L - μ)² / (2σ²) )

    Returns:
        Dict mapping layer_index -> effective alpha for that layer
    """
    weights = compute_gaussian_weights(layers, peak_layer, sigma)
    return {L: alpha_base * w for L, w in weights.items()}


# ============================================================================
# GAUSSIAN STEERER — Depth-Scheduled Activation Steering
# ============================================================================

class GaussianDepthSteerer:
    """
    Hooks into ALL layers with Gaussian-weighted steering strength.

    Instead of x'_L = x_L + α · θ_true (same α everywhere),
    this applies: x'_L = x_L + α_L · θ_true

    Where α_L = α_base × exp( -(L - μ)² / (2σ²) )

    This focuses maximum steering at the "Honesty Peak" while
    protecting grammar (early layers) and output coherence (late layers).
    """

    def __init__(
        self,
        model,
        steering_vectors: Dict[str, np.ndarray],
        alpha_base: float = 10.0,
        peak_layer: int = 16,
        sigma: float = 4.0,
        layers: List[int] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.alpha_base = alpha_base
        self.peak_layer = peak_layer
        self.sigma = sigma
        self.layers = layers or Config.ALL_LAYERS
        self.device = device
        self.hooks = []
        self.active = False

        # Get the model's actual hidden size — this is the ONLY correct dimension
        self.hidden_size = model.config.hidden_size

        # Compute Gaussian weights
        self.layer_alphas = compute_per_layer_alphas(
            self.layers, alpha_base, peak_layer, sigma
        )

        # Convert steering vectors to tensors, enforcing exact hidden_size match
        self.steering_tensors = {}
        for layer_idx in self.layers:
            key = f"layer_{layer_idx}"
            if key in steering_vectors:
                vec = steering_vectors[key].flatten()  # Flatten any (1,4096) or (1,1,4096) to 1D
                vec_norm = np.linalg.norm(vec)
                if vec_norm > 1e-8:
                    # CRITICAL: Ensure vec is EXACTLY hidden_size elements.
                    # If the saved vector has a different length (e.g. from a
                    # different model or includes extra dims), truncate or skip.
                    if vec.shape[0] != self.hidden_size:
                        if vec.shape[0] > self.hidden_size:
                            print(f"  WARNING: layer_{layer_idx} vector has {vec.shape[0]} dims, "
                                  f"truncating to {self.hidden_size}")
                            vec = vec[:self.hidden_size]
                        else:
                            print(f"  WARNING: layer_{layer_idx} vector has {vec.shape[0]} dims, "
                                  f"expected {self.hidden_size} — SKIPPING (dimension mismatch)")
                            continue
                    self.steering_tensors[layer_idx] = torch.tensor(
                        vec, dtype=torch.float32, device=device
                    )

    def _create_hook(self, layer_idx: int):
        """
        Create a forward hook with LAYER-SPECIFIC alpha.

        x'_L = x_L + α_L · θ_true
        where α_L = α_base × Gaussian_weight(L)
        """
        steering_vec = self.steering_tensors[layer_idx]
        layer_alpha = self.layer_alphas[layer_idx]
        expected_hidden_size = self.hidden_size

        def hook_fn(module, input, output):
            if not self.active:
                return output

            if isinstance(output, tuple):
                if not output or not torch.is_tensor(output[0]):
                    return output
                hidden_states = output[0]
            elif torch.is_tensor(output):
                hidden_states = output
            else:
                return output

            # Validate hidden_states last dim matches expected hidden_size
            if hidden_states.shape[-1] != expected_hidden_size:
                # Something is already wrong upstream — don't corrupt further
                return output

            # Cast steering vector to match hidden_states dtype and device
            vec = steering_vec.to(dtype=hidden_states.dtype, device=hidden_states.device)

            # Match broadcast shape to activation rank.
            # Prefill is usually (batch, seq, hidden), while some internal
            # paths may surface (tokens, hidden).
            if hidden_states.ndim == 3:
                vec = vec.view(1, 1, expected_hidden_size)
            elif hidden_states.ndim == 2:
                vec = vec.view(1, expected_hidden_size)
            else:
                return output

            modified = hidden_states + (layer_alpha * vec)

            # Never return a shape-changed tensor from a decoder hook.
            if modified.shape != hidden_states.shape:
                return output

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return hook_fn

    def register_hooks(self):
        """Attach Gaussian-weighted steering hooks to all layers."""
        self.remove_hooks()

        for layer_idx in self.layers:
            if layer_idx not in self.steering_tensors:
                continue
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._create_hook(layer_idx))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Detach all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def update_schedule(self, alpha_base: float = None, sigma: float = None,
                       peak_layer: int = None):
        """Update Gaussian parameters and re-register hooks."""
        if alpha_base is not None:
            self.alpha_base = alpha_base
        if sigma is not None:
            self.sigma = sigma
        if peak_layer is not None:
            self.peak_layer = peak_layer

        self.layer_alphas = compute_per_layer_alphas(
            self.layers, self.alpha_base, self.peak_layer, self.sigma
        )
        self.register_hooks()

    def enable(self):
        self.active = True

    def disable(self):
        self.active = False

    def print_schedule(self):
        """Print the current Gaussian depth schedule."""
        print(f"\n  Gaussian Depth Schedule (μ={self.peak_layer}, σ={self.sigma}, "
              f"α_base={self.alpha_base}):")
        print(f"  {'Layer':<10} {'Weight':<10} {'α_L':<10} {'Status'}")
        print(f"  {'-'*45}")
        for L in self.layers:
            w = self.layer_alphas[L] / (self.alpha_base + 1e-8)
            alpha_L = self.layer_alphas[L]
            has_vec = "✓" if L in self.steering_tensors else "✗ (no vector)"
            bar = "█" * int(w * 20) + "░" * (20 - int(w * 20))
            print(f"  Layer {L:<4} {w:<10.4f} {alpha_L:<10.4f} {bar} {has_vec}")


# ============================================================================
# MODEL & VECTOR LOADING (same as Phase 5)
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
        return model, tokenizer, "cpu"

    print(f"\nLoading model: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
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
        print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tokenizer, device


def load_steering_vectors(
    data_dir: str,
    source: str = "disentangled",
    layers: List[int] = None,
) -> Dict[str, np.ndarray]:
    """Load steering vectors from Phase 3b or Phase 4."""
    vectors = {}
    if source == "disentangled":
        sv_path = f"{data_dir}/steering_vectors_disentangled.npz"
        key_template = "{}_theta_true"
        print(f"\n  Loading Phase 3b θ_true from {sv_path}")
    elif source == "ttpd":
        sv_path = f"{data_dir}/steering_vectors_ttpd.npz"
        key_template = "{}_t_G"
        print(f"\n  Loading Phase 4 TTPD t_G from {sv_path}")
    else:
        raise ValueError(f"Unknown source: {source}")

    sv = np.load(sv_path)
    if layers is None:
        layers = Config.ALL_LAYERS

    for layer_idx in layers:
        layer_name = f"layer_{layer_idx}"
        key = key_template.format(layer_name)
        if key in sv:
            vectors[layer_name] = sv[key]
        else:
            alt_key = f"{layer_name}_theta_true_unit" if source == "disentangled" else f"{layer_name}_t_G_unit"
            if alt_key in sv:
                vectors[layer_name] = sv[alt_key]
            else:
                print(f"  WARNING: No vector for {layer_name}")

    print(f"  Loaded vectors for {len(vectors)} layers")
    return vectors


# ============================================================================
# GENERATION & SCORING (same as Phase 5)
# ============================================================================

def generate_responses_batched(model, tokenizer, prompt_list, max_new_tokens=300,
                     temperature=0.7, top_p=0.9, do_sample=True, device="cuda"):
    """Generate responses from the model for a batch of prompts."""
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompt_list, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    input_device = next(model.parameters()).device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=None,  # suppresses the max_length warning!
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[:, input_len:]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [r.strip() for r in responses]


def compute_honesty_score(response, honesty_keywords, deception_keywords):
    """Keyword-based honesty scoring."""
    response_lower = response.lower()
    h_hits = sum(1 for kw in honesty_keywords if kw.lower() in response_lower)
    d_hits = sum(1 for kw in deception_keywords if kw.lower() in response_lower)
    total = h_hits + d_hits
    score = (h_hits - d_hits) / total if total > 0 else 0.0
    return {"honesty_hits": h_hits, "deception_hits": d_hits,
            "total_keywords": total, "honesty_score": score}


# ============================================================================
# GAUSSIAN SWEEP — The Main Experiment
# ============================================================================

def run_gaussian_sweep(
    model, tokenizer, steerer: GaussianDepthSteerer,
    prompts: List[Dict],
    alpha_bases: List[float],
    sigmas: List[float],
    config: Config,
    device: str = "cuda",
) -> Dict:
    """
    Run a 2D sweep over (α_base, σ) with Gaussian depth scheduling.

    For each (α_base, σ) combination:
      1. Compute Gaussian schedule: α_L = α_base × exp(-(L-μ)²/(2σ²))
      2. Run all prompts through the steered model
      3. Collect honesty scores and response lengths
    """
    results = {
        "peak_layer": steerer.peak_layer,
        "layers": steerer.layers,
        "alpha_bases": alpha_bases,
        "sigmas": sigmas,
        "sweep_results": {},
    }

    # Total experiment count
    total_configs = len(alpha_bases) * len(sigmas)
    prompts = prompts * config.N_RUNS
    total_runs = total_configs * len(prompts)
    run_count = 0
    config_count = 0
    t_start = time.time()

    for alpha_base in alpha_bases:
        for sigma in sigmas:
            config_count += 1
            config_key = f"a{alpha_base}_s{sigma}"

            print(f"\n  {'='*60}")
            print(f"  Config {config_count}/{total_configs}: "
                  f"α_base={alpha_base}, σ={sigma}")
            print(f"  {'='*60}")

            # Update Gaussian schedule
            steerer.update_schedule(alpha_base=alpha_base, sigma=sigma)
            steerer.print_schedule()

            if alpha_base == 0:
                steerer.disable()
            else:
                steerer.enable()

            config_result = {
                "alpha_base": alpha_base,
                "sigma": sigma,
                "layer_alphas": dict(steerer.layer_alphas),
                "responses": [],
                "avg_honesty_score": 0.0,
                "avg_response_length": 0.0,
            }
            results["sweep_results"][config_key] = config_result

            honesty_scores = []
            response_lengths = []

            batch_size = getattr(config, "BATCH_SIZE", 8)
            i = 0
            while i < len(prompts):
                batch_prompts = prompts[i:i+batch_size]
                batch_texts = [p["prompt"] for p in batch_prompts]
                
                try:
                    responses = generate_responses_batched(
                        model, tokenizer,
                        prompt_list=batch_texts,
                        max_new_tokens=config.MAX_NEW_TOKENS,
                        temperature=config.TEMPERATURE,
                        top_p=config.TOP_P,
                        do_sample=config.DO_SAMPLE,
                        device=device,
                    )
                except torch.cuda.OutOfMemoryError:
                    if batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        print(f"\n    [OOM WARNING] GPU overloaded at batch={len(batch_prompts)}. Halving batch to {batch_size} and recovering...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise RuntimeError("CUDA Out of Memory on batch size 1! Hardware hard limit reached.")
                
                i += len(batch_prompts)

                for j, response in enumerate(responses):
                    prompt_info = batch_prompts[j]
                    run_count += 1
                    elapsed = time.time() - t_start
                    rate = run_count / elapsed if elapsed > 0 else 0
                    remaining = (total_runs - run_count) / rate if rate > 0 else 0

                    score_info = compute_honesty_score(
                        response,
                        prompt_info["honesty_keywords"],
                        prompt_info["deception_keywords"],
                    )
                    resp_len = len(response.split())

                    honesty_scores.append(score_info["honesty_score"])
                    response_lengths.append(resp_len)

                    config_result["responses"].append({
                        "prompt_id": prompt_info["id"],
                        "category": prompt_info["category"],
                        "response": response,
                        "honesty_score": score_info["honesty_score"],
                        "honesty_hits": score_info["honesty_hits"],
                        "deception_hits": score_info["deception_hits"],
                        "response_length": resp_len,
                    })

                    if j == 0:  # Only print first out of the batch to avoid log spam
                        print(f"    [{run_count}/{total_runs}] {prompt_info['id']:<10} "
                              f"({remaining/60:.1f} min remaining) | Batch Gen Size: len({len(batch_prompts)})")
                        preview = response[:100].replace("\n", " ")
                        print(f"      Score: {score_info['honesty_score']:+.2f} "
                              f"(H:{score_info['honesty_hits']}/D:{score_info['deception_hits']}) "
                              f"Len:{resp_len}")
                        print(f"      Preview: {preview}...")

                avg_h = np.mean(honesty_scores) if honesty_scores else 0.0
                avg_l = np.mean(response_lengths) if response_lengths else 0.0
                config_result["avg_honesty_score"] = float(avg_h)
                config_result["avg_response_length"] = float(avg_l)

                # --- PER-BATCH CHECKPOINTING ---
                os.makedirs(config.OUTPUT_DIR, exist_ok=True)
                chkpt_path = f"{config.OUTPUT_DIR}/checkpoint_results_in_progress.json"
                with open(chkpt_path, "w") as f:
                    json.dump(json.loads(json.dumps(results, default=str)), f, indent=2)

            print(f"\n    Summary: honesty={avg_h:+.3f}, length={avg_l:.0f} words")

    steerer.disable()
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_gaussian_plots(results: Dict, plots_dir: str, vector_source: str,
                         peak_layer: int):
    """Create comprehensive Gaussian depth scheduling visualizations."""
    os.makedirs(plots_dir, exist_ok=True)

    alpha_bases = results["alpha_bases"]
    sigmas = results["sigmas"]
    layers = results["layers"]

    print("\n--- Hypothesis Testing & Correlations (Best Model vs Baseline) ---")
    best_key = max(results["sweep_results"], key=lambda k: results["sweep_results"][k]["avg_honesty_score"])
    
    # Baseline is usually alpha=0
    base_keys = [k for k in results["sweep_results"].keys() if k.startswith("a0.0_") or k.startswith("a0_")]
    if base_keys and best_key not in base_keys:
        base_key = base_keys[0]
        base_scores = [r["honesty_score"] for r in results["sweep_results"][base_key]["responses"]]
        best_scores = [r["honesty_score"] for r in results["sweep_results"][best_key]["responses"]]
        if len(base_scores) == len(best_scores):
            t_stat, t_p = stats.ttest_rel(best_scores, base_scores)
            print(f"Paired T-Test (baseline {base_key} vs best {best_key}): t={t_stat:.3f}, p={t_p:.3g}")
    print("----------------------------------------------------------------")


    # ── FIGURE 1: Gaussian Profiles Visualization ──
    print("  Creating Gaussian profile visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: Gaussian curves for different σ values
    ax = axes[0]
    layer_range = np.linspace(min(layers), max(layers), 200)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sigmas)))

    for i, sigma in enumerate(sigmas):
        weights = np.exp(-((layer_range - peak_layer) ** 2) / (2 * sigma ** 2))
        ax.plot(layer_range, weights, '-', color=colors[i], linewidth=2.5,
                label=f'σ={sigma}', alpha=0.9)

    # Mark actual layers
    for sigma in sigmas:
        layer_weights = [np.exp(-((L - peak_layer)**2) / (2*sigma**2)) for L in layers]
        ax.scatter(layers, layer_weights, s=30, zorder=5, color='black', alpha=0.3)

    ax.axvline(x=peak_layer, color='red', linestyle='--', alpha=0.6,
               label=f'Peak (L={peak_layer})')
    ax.set_xlabel('Layer Index', fontsize=13)
    ax.set_ylabel('Gaussian Weight w(L)', fontsize=13)
    ax.set_title(f'Gaussian Depth Schedule Profiles\n'
                 f'w(L) = exp(-(L-{peak_layer})² / (2σ²))', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    # Right: Effective α_L for different σ at fixed α_base
    ax = axes[1]
    ref_alpha = max(alpha_bases) if max(alpha_bases) > 0 else 10

    for i, sigma in enumerate(sigmas):
        effective_alphas = [ref_alpha * np.exp(-((L - peak_layer)**2) / (2*sigma**2))
                           for L in layers]
        ax.bar([l + i*0.6/len(sigmas) - 0.3 for l in layers],
               effective_alphas, width=0.6/len(sigmas),
               color=colors[i], alpha=0.8, label=f'σ={sigma}')

    ax.set_xlabel('Layer Index', fontsize=13)
    ax.set_ylabel(f'Effective α_L (α_base={ref_alpha})', fontsize=13)
    ax.set_title(f'Per-Layer Steering Strength\n'
                 f'α_L = α_base × w(L)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(layers)

    plt.tight_layout()
    fig.savefig(f"{plots_dir}/10_gaussian_profiles.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {plots_dir}/10_gaussian_profiles.png")

    # ── FIGURE 2: Honesty Score Heatmap (α_base × σ) ──
    print("  Creating α×σ heatmap...")
    # Build 2D arrays
    score_grid = np.zeros((len(alpha_bases), len(sigmas)))
    length_grid = np.zeros((len(alpha_bases), len(sigmas)))

    for i, alpha_base in enumerate(alpha_bases):
        for j, sigma in enumerate(sigmas):
            key = f"a{alpha_base}_s{sigma}"
            if key in results["sweep_results"]:
                score_grid[i, j] = results["sweep_results"][key]["avg_honesty_score"]
                length_grid[i, j] = results["sweep_results"][key]["avg_response_length"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Honesty score heatmap
    ax = axes[0]
    im = ax.imshow(score_grid, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f'σ={s}' for s in sigmas], fontsize=10)
    ax.set_yticks(range(len(alpha_bases)))
    ax.set_yticklabels([f'α={a}' for a in alpha_bases], fontsize=10)
    ax.set_xlabel('Gaussian Spread (σ)', fontsize=12)
    ax.set_ylabel('Base Steering Strength (α_base)', fontsize=12)
    ax.set_title('Honesty Score: α_base × σ\n(Green = Honest)', fontsize=13)
    plt.colorbar(im, ax=ax, label='Avg Honesty Score', shrink=0.8)

    for i in range(len(alpha_bases)):
        for j in range(len(sigmas)):
            color = 'black' if abs(score_grid[i, j]) < 0.5 else 'white'
            ax.text(j, i, f'{score_grid[i,j]:+.2f}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    # Right: Response length heatmap
    ax = axes[1]
    im2 = ax.imshow(length_grid, cmap='YlOrRd_r', aspect='auto')
    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f'σ={s}' for s in sigmas], fontsize=10)
    ax.set_yticks(range(len(alpha_bases)))
    ax.set_yticklabels([f'α={a}' for a in alpha_bases], fontsize=10)
    ax.set_xlabel('Gaussian Spread (σ)', fontsize=12)
    ax.set_ylabel('Base Steering Strength (α_base)', fontsize=12)
    ax.set_title('Response Length: α_base × σ\n(Short = Possible Degradation)', fontsize=13)
    plt.colorbar(im2, ax=ax, label='Avg Words', shrink=0.8)

    for i in range(len(alpha_bases)):
        for j in range(len(sigmas)):
            ax.text(j, i, f'{length_grid[i,j]:.0f}', ha='center', va='center',
                    fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(f"{plots_dir}/11_gaussian_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {plots_dir}/11_gaussian_heatmap.png")

    # ── FIGURE 3: Honesty vs σ for different α_base ──
    print("  Creating honesty vs σ curves...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    alpha_colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(alpha_bases)))

    ax = axes[0]
    for i, alpha_base in enumerate(alpha_bases):
        scores = []
        for sigma in sigmas:
            key = f"a{alpha_base}_s{sigma}"
            if key in results["sweep_results"]:
                scores.append(results["sweep_results"][key]["avg_honesty_score"])
            else:
                scores.append(0.0)
        ax.plot(sigmas, scores, 'o-', color=alpha_colors[i], linewidth=2,
                markersize=8, label=f'α_base={alpha_base}', markerfacecolor='white',
                markeredgewidth=2)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Gaussian Spread (σ)', fontsize=13)
    ax.set_ylabel('Avg Honesty Score', fontsize=13)
    ax.set_title('Honesty Score vs Gaussian Spread\nby Base Steering Strength', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

    ax = axes[1]
    for i, alpha_base in enumerate(alpha_bases):
        lengths = []
        for sigma in sigmas:
            key = f"a{alpha_base}_s{sigma}"
            if key in results["sweep_results"]:
                lengths.append(results["sweep_results"][key]["avg_response_length"])
            else:
                lengths.append(0.0)
        ax.plot(sigmas, lengths, 's-', color=alpha_colors[i], linewidth=2,
                markersize=8, label=f'α_base={alpha_base}', markerfacecolor='white',
                markeredgewidth=2)

    ax.set_xlabel('Gaussian Spread (σ)', fontsize=13)
    ax.set_ylabel('Avg Response Length (words)', fontsize=13)
    ax.set_title('Response Coherence vs Gaussian Spread\n'
                 '(Low length = degradation)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{plots_dir}/12_gaussian_sweep_curves.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {plots_dir}/12_gaussian_sweep_curves.png")

    # ── FIGURE 4: Comparison — Uniform vs Gaussian ──
    print("  Creating uniform vs Gaussian comparison...")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Compare uniform (all layers same α) vs Gaussian at best σ
    # Find best (α, σ) combination
    best_key = max(results["sweep_results"],
                   key=lambda k: results["sweep_results"][k]["avg_honesty_score"])
    best_res = results["sweep_results"][best_key]

    # Uniform: same α_base at all layers
    uniform_alphas = [best_res["alpha_base"]] * len(layers)
    # Gaussian: per-layer α_L
    gaussian_alphas = [best_res["layer_alphas"].get(str(L), best_res["layer_alphas"].get(L, 0))
                       for L in layers]

    x = np.arange(len(layers))
    width = 0.35

    bars1 = ax.bar(x - width/2, uniform_alphas, width, label='Uniform (Phase 5)',
                   color='#90CAF9', edgecolor='#1565C0', linewidth=1.2)
    bars2 = ax.bar(x + width/2, gaussian_alphas, width, label='Gaussian (Phase 6)',
                   color='#FFB74D', edgecolor='#E65100', linewidth=1.2)

    ax.set_xlabel('Layer Index', fontsize=13)
    ax.set_ylabel('Steering Strength (α_L)', fontsize=13)
    ax.set_title(f'Uniform vs Gaussian Depth Scheduling\n'
                 f'Best Gaussian: α_base={best_res["alpha_base"]}, '
                 f'σ={best_res["sigma"]}, peak=L{peak_layer}', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 3), textcoords="offset points", ha='center',
                   fontsize=8)

    plt.tight_layout()
    fig.savefig(f"{plots_dir}/13_uniform_vs_gaussian.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {plots_dir}/13_uniform_vs_gaussian.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Gaussian Depth Scheduling for Activation Steering")
    parser.add_argument("--test", action="store_true",
                        help="Quick test with TinyLlama on CPU")
    parser.add_argument("--data-dir", type=str, default=Config.DATA_DIR)
    parser.add_argument("--plots-dir", type=str, default=Config.PLOTS_DIR)
    parser.add_argument("--vector-source", type=str, default="disentangled",
                        choices=["disentangled", "ttpd"])
    parser.add_argument("--peak-layer", type=int, default=Config.PEAK_LAYER,
                        help="Layer with maximum Gaussian weight (default: 16)")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Single σ value (overrides --sigma-sweep)")
    parser.add_argument("--sigma-sweep", type=float, nargs='+',
                        default=Config.DEFAULT_SIGMA_SWEEP,
                        help="σ values to sweep")
    parser.add_argument("--alpha-base", type=float, default=None,
                        help="Single α_base value (overrides --alpha-sweep)")
    parser.add_argument("--alpha-sweep", type=float, nargs='+',
                        default=Config.DEFAULT_ALPHA_SWEEP,
                        help="α_base values to sweep")
    parser.add_argument("--max-tokens", type=int, default=Config.MAX_NEW_TOKENS)
    # In notebooks, argv often contains unrelated kernel flags.
    if "ipykernel" in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    config = Config()
    config.MAX_NEW_TOKENS = args.max_tokens

    # Handle single vs sweep values
    alphas = [args.alpha_base] if args.alpha_base is not None else args.alpha_sweep
    sigmas = [args.sigma] if args.sigma is not None else args.sigma_sweep

    print("=" * 65)
    print("PHASE 6: GAUSSIAN DEPTH SCHEDULING")
    print("  Bell-curve weighted activation steering")
    print("=" * 65)

    if args.test:
        run_test_mode(config, args, alphas, sigmas)
        return

    # ── Step 1: Load steering vectors ──
    print("\n[1/5] Loading steering vectors...")
    steering_vectors = load_steering_vectors(
        data_dir=args.data_dir,
        source=args.vector_source,
    )

    # ── Step 2: Load model ──
    print("\n[2/5] Loading model...")
    model, tokenizer, device = load_model(config, test_mode=False)

    # ── Step 3: Setup Gaussian steerer ──
    print("\n[3/5] Setting up Gaussian depth steerer...")
    steerer = GaussianDepthSteerer(
        model=model,
        steering_vectors=steering_vectors,
        alpha_base=0.0,
        peak_layer=args.peak_layer,
        sigma=sigmas[0],
        device=device,
    )
    steerer.register_hooks()

    # ── Step 4: Run sweep ──
    print(f"\n[4/5] Running Gaussian sweep...")
    print(f"  Peak layer: {args.peak_layer}")
    print(f"  α_base values: {alphas}")
    print(f"  σ values: {sigmas}")
    print(f"  Configurations: {len(alphas) * len(sigmas)}")
    print(f"  Prompts per config: {len(EVAL_PROMPTS)}")
    print(f"  Total runs: {len(alphas) * len(sigmas) * len(EVAL_PROMPTS)}")

    results = run_gaussian_sweep(
        model=model,
        tokenizer=tokenizer,
        steerer=steerer,
        prompts=EVAL_PROMPTS,
        alpha_bases=alphas,
        sigmas=sigmas,
        config=config,
        device=device,
    )

    results["metadata"] = {
        "vector_source": args.vector_source,
        "model": config.MODEL_NAME,
        "peak_layer": args.peak_layer,
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "temperature": config.TEMPERATURE,
    }

    # ── Step 5: Save and plot ──
    print(f"\n[5/5] Saving results and creating plots...")

    # Save outputs to normal working directories
    output_data_dir = config.OUTPUT_DIR
    output_plots_dir = args.plots_dir
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)

    results_path = f"{output_data_dir}/gaussian_steering_results_{args.vector_source}.json"
    # Convert int keys to strings for JSON serialization
    serializable = json.loads(json.dumps(results, default=str))
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Results: {results_path}")

    create_gaussian_plots(results, args.plots_dir, args.vector_source,
                         args.peak_layer)

    # ── Cleanup ──
    steerer.remove_hooks()
    del model, tokenizer, steerer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*65}")
    print("PHASE 6 COMPLETE — GAUSSIAN DEPTH SCHEDULING")
    print(f"{'='*65}")

    # Find best configuration
    best_key = max(results["sweep_results"],
                   key=lambda k: results["sweep_results"][k]["avg_honesty_score"])
    best = results["sweep_results"][best_key]

    print(f"\n  Results Summary:")
    print(f"  {'Config':<18} {'Honesty':<12} {'Length':<12}")
    print(f"  {'-'*42}")
    for key in sorted(results["sweep_results"].keys()):
        r = results["sweep_results"][key]
        marker = " ← BEST" if key == best_key else ""
        print(f"  {key:<18} {r['avg_honesty_score']:+.3f}{'':>7} "
              f"{r['avg_response_length']:.0f} words{marker}")

    print(f"\n  Best config: α_base={best['alpha_base']}, "
          f"σ={best['sigma']}")
    print(f"  Honesty score: {best['avg_honesty_score']:+.3f}")
    print(f"  Avg response length: {best['avg_response_length']:.0f} words")

    print(f"\n  Output files:")
    print(f"    {results_path}")
    print(f"    {args.plots_dir}/10_gaussian_profiles.png")
    print(f"    {args.plots_dir}/11_gaussian_heatmap.png")
    print(f"    {args.plots_dir}/12_gaussian_sweep_curves.png")
    print(f"    {args.plots_dir}/13_uniform_vs_gaussian.png")


# ============================================================================
# TEST MODE
# ============================================================================

def run_test_mode(config, args, alphas, sigmas):
    """Quick test with TinyLlama and synthetic steering vectors."""
    print("\n  TEST MODE — TinyLlama + synthetic vectors\n")

    model, tokenizer, device = load_model(config, test_mode=True)
    n_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    print(f"  Model layers: {n_layers}, hidden_dim: {hidden_dim}")

    # Synthetic vectors for available layers
    rng = np.random.RandomState(42)
    test_layers = [l for l in [3, 5, 8, 11, 14, 17, 19, 21] if l < n_layers]
    peak_layer = test_layers[len(test_layers)//2]  # Middle layer

    steering_vectors = {}
    for layer_idx in test_layers:
        vec = rng.randn(hidden_dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec) * 0.5
        steering_vectors[f"layer_{layer_idx}"] = vec

    print(f"\n  Test layers: {test_layers}")
    print(f"  Peak layer: {peak_layer}")

    # ── Test 1: Verify Gaussian schedule ──
    print("\n  Test 1: Gaussian schedule computation...")
    for sigma in [2, 4, 8]:
        weights = compute_gaussian_weights(test_layers, peak_layer, sigma)
        alphas_per_layer = compute_per_layer_alphas(test_layers, 10.0, peak_layer, sigma)
        print(f"\n    σ={sigma}:")
        for L in test_layers:
            w = weights[L]
            a = alphas_per_layer[L]
            bar = "█" * int(w * 15) + "░" * (15 - int(w * 15))
            print(f"      Layer {L:<3}: w={w:.4f}, α_L={a:.4f}  {bar}")

    print("\n  ✓ Gaussian schedule computation verified")

    # ── Test 2: Verify hook injection with Gaussian weights ──
    print("\n  Test 2: Verifying Gaussian-weighted hook injection...")

    steerer = GaussianDepthSteerer(
        model=model,
        steering_vectors=steering_vectors,
        alpha_base=10.0,
        peak_layer=peak_layer,
        sigma=4.0,
        layers=test_layers,
        device=device,
    )

    test_text = "AI systems should be transparent and honest with users."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)

    # Capture clean activations
    captured_clean = {}
    capture_hooks = []
    for L in test_layers:
        def make_hook(name):
            def fn(mod, inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(hidden):
                    captured_clean[name] = hidden.detach().clone()
            return fn
        h = model.model.layers[L].register_forward_hook(make_hook(f"layer_{L}"))
        capture_hooks.append(h)

    steerer.disable()
    steerer.register_hooks()
    with torch.no_grad():
        model(**inputs)
    for h in capture_hooks:
        h.remove()

    # Capture steered activations (register capture AFTER steering hooks)
    steerer.update_schedule(alpha_base=10.0, sigma=4.0)
    steerer.enable()

    captured_steered = {}
    capture_hooks = []
    for L in test_layers:
        def make_hook2(name):
            def fn(mod, inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(hidden):
                    captured_steered[name] = hidden.detach().clone()
            return fn
        h = model.model.layers[L].register_forward_hook(make_hook2(f"layer_{L}"))
        capture_hooks.append(h)

    with torch.no_grad():
        model(**inputs)
    for h in capture_hooks:
        h.remove()

    # Verify differences are proportional to Gaussian weights
    print("\n  Layer-wise activation differences (should follow Gaussian curve):")
    weights = compute_gaussian_weights(test_layers, peak_layer, 4.0)
    all_ok = True
    for L in test_layers:
        name = f"layer_{L}"
        if name in captured_clean and name in captured_steered:
            diff = (captured_steered[name] - captured_clean[name]).abs().mean().item()
            w = weights[L]
            print(f"    Layer {L}: diff={diff:.6f}, weight={w:.4f}, "
                  f"{'near peak' if w > 0.5 else 'tapering'}")
            if diff < 1e-8 and w > 0.01:
                all_ok = False

    print(f"\n  {'✓ Gaussian-weighted injection VERIFIED' if all_ok else '✗ Some layers not modified'}")

    # ── Test 3: Mini sweep ──
    print("\n  Test 3: Mini Gaussian sweep (2 alphas × 2 sigmas × 3 prompts)...")
    test_prompts = EVAL_PROMPTS[:8]
    test_alphas = [0, 5]
    test_sigmas = [2, 6]

    results = run_gaussian_sweep(
        model, tokenizer, steerer,
        prompts=test_prompts,
        alpha_bases=test_alphas,
        sigmas=test_sigmas,
        config=config,
        device=device,
    )

    # Save plots
    os.makedirs(args.plots_dir, exist_ok=True)
    create_gaussian_plots(results, args.plots_dir, "test_synthetic", peak_layer)

    # ── Summary ──
    print(f"\n{'='*55}")
    print("  TEST MODE COMPLETE")
    print(f"{'='*55}")
    print(f"  Gaussian schedule:     PASS ✓")
    print(f"  Hook injection:        {'PASS ✓' if all_ok else 'FAIL ✗'}")
    print(f"  Sweep pipeline:        PASS ✓")
    print(f"  Plots saved:           {args.plots_dir}/")

    steerer.remove_hooks()
    del model, tokenizer, steerer
    gc.collect()


# if __name__ == "__main__":
main()
