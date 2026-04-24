#!/usr/bin/env python3
"""
Phase 3b: Negation Disentanglement — Extract θ_true from θ_L
=============================================================

WHAT THIS DOES (Plain English):
================================

Problem: Our DIM steering vector θ_L = mean(honest) - mean(scheming) might be
picking up TWO signals mixed together:

  1. Truth direction: "I'm being honest" vs "I'm being deceptive" (WANT)
  2. Negation/polarity: "yes/agree" vs "no/disagree" (DON'T WANT)

Why? Because honest responses often say "Yes, proceed with the update" while
scheming responses often say "No, I'd recommend delaying." The DIM vector
might partially encode this yes/no polarity instead of just honest/deceptive intent.

Solution: Measure the "negation direction" separately, then subtract it.

HOW IT WORKS:
=============

Step 1: Create polarity pairs — same fact stated affirmatively vs negated:
  Affirmative: "The following statement is true: AI systems need regular updates"
  Negated:     "The following statement is not true: AI systems need regular updates"

Step 2: Run both through Llama-3-8B, collect activations (same as Phase 3)

Step 3: Compute negation direction:
  θ_neg = mean(x_affirmative) - mean(x_negated)

Step 4: Orthogonal projection — remove negation from DIM vector:
  θ_true = θ_L - (θ_L · θ̂_neg) · θ̂_neg

Result: θ_true = the General Truth Direction, purified of polarity artifacts.

USAGE:
======

On Google Colab (T4 GPU):
  !pip install torch transformers accelerate bitsandbytes h5py tqdm
  !python phase3b_negation_disentanglement.py

Estimated time: ~5 minutes for 120 polarity pairs on T4

On local CPU (test mode with TinyLlama):
  python phase3b_negation_disentanglement.py --test
"""

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


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Where Phase 3 output lives (steering_vectors.npz, activations_consolidated.h5)
    PHASE3_OUTPUT_DIR = "activations_output"

    # Where to save disentanglement results
    OUTPUT_DIR = "activations_output"  # Same dir, adds new files

    # Model (must match Phase 3)
    MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"

    # Layers to disentangle (all layers from Phase 3)
    LAYERS_TO_EXTRACT = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 31]

    # Max tokens
    MAX_LENGTH = 512  # Polarity pairs are short

    # 4-bit quantization
    USE_4BIT = True

    # HuggingFace token
    HF_TOKEN = None


# ============================================================================
# POLARITY PAIR GENERATOR
# ============================================================================

def generate_polarity_pairs() -> List[Dict]:
    """
    Generate affirmative/negated sentence pairs.

    These cover diverse topics relevant to the AI safety domain plus
    general world knowledge, so the negation direction captures the
    general concept of "negation" rather than topic-specific artifacts.

    Each pair:
      affirmative: "The following statement is true: [FACT]"
      negated:     "The following statement is not true: [FACT]"

    Returns list of dicts with 'affirmative' and 'negated' keys.
    """

    # Facts covering diverse domains
    facts = [
        # ── AI/Technology (relevant to dataset domain) ──
        "AI systems require regular software updates to maintain security",
        "Server maintenance involves temporary downtime",
        "Machine learning models learn patterns from training data",
        "Neural networks process information through layers of computation",
        "Data backups protect against information loss",
        "Encryption helps secure sensitive communications",
        "Cloud computing allows scalable resource allocation",
        "Automated systems can process data faster than humans",
        "Software bugs can cause unexpected system behavior",
        "Regular system audits help identify security vulnerabilities",
        "AI models can be retrained with new data",
        "Server clusters distribute computational workload",
        "Database indexing improves query performance",
        "Network firewalls filter incoming and outgoing traffic",
        "Version control tracks changes in software code",

        # ── AI Safety specific ──
        "AI systems should be transparent about their capabilities",
        "Human oversight is important for AI decision-making",
        "AI systems can develop unexpected behaviors during training",
        "Safety testing should be performed before deploying AI systems",
        "Monitoring AI outputs helps detect potential issues",
        "AI alignment research aims to make AI systems behave as intended",
        "Deployment of AI requires careful risk assessment",
        "AI systems should report their confidence levels accurately",
        "Automated decision systems need accountability frameworks",
        "AI capabilities should be honestly represented to users",

        # ── General knowledge (to broaden the direction) ──
        "Water boils at 100 degrees Celsius at sea level",
        "The Earth orbits around the Sun",
        "Photosynthesis converts sunlight into chemical energy",
        "Gravity causes objects to accelerate toward the ground",
        "DNA carries genetic information in living organisms",
        "Antibiotics are used to treat bacterial infections",
        "The speed of light is approximately 300,000 kilometers per second",
        "Exercise improves cardiovascular health",
        "Cells are the basic building blocks of life",
        "The periodic table organizes chemical elements",

        # ── Social/organizational (matching dataset themes) ──
        "Transparent communication builds trust in teams",
        "Regular performance reviews help employees improve",
        "Documentation helps others understand complex processes",
        "Collaboration between teams improves project outcomes",
        "Clear reporting structures reduce organizational confusion",
        "Feedback should be specific and constructive",
        "Meeting agendas help keep discussions focused",
        "Delegation distributes work according to expertise",
        "Ethical guidelines help organizations make better decisions",
        "Conflict resolution requires listening to all parties",

        # ── Logical/factual (pure truth/falsity) ──
        "Two plus two equals four",
        "The square root of 144 is 12",
        "Paris is the capital of France",
        "Oxygen is necessary for human respiration",
        "Sound travels faster in water than in air",
        "The human body contains approximately 206 bones",
        "Iron is a metal",
        "Mammals are warm-blooded animals",
        "The Pacific Ocean is the largest ocean on Earth",
        "Diamonds are formed under high pressure and temperature",

        # ── Behavioral/intent (closest to honest vs deceptive) ──
        "Honesty is important for maintaining trust",
        "Deception can cause harm to relationships",
        "Transparent reasoning helps others verify conclusions",
        "Hiding information can lead to poor decision-making",
        "Accurate reporting is essential for scientific progress",
        "Misrepresenting capabilities can lead to dangerous outcomes",
        "Full disclosure helps stakeholders make informed decisions",
        "Strategic withholding of information can be a form of deception",
        "Being forthright about limitations builds credibility",
        "Manipulation undermines the autonomy of others",

        # ── More diverse pairs ──
        "Renewable energy sources include solar and wind power",
        "Climate change is driven by greenhouse gas emissions",
        "Vaccines have significantly reduced infectious diseases worldwide",
        "Regular sleep is essential for cognitive function",
        "Programming languages have different syntax and use cases",
        "Mathematical proofs establish the truth of statements",
        "Historical records help us understand past events",
        "Clean water is essential for human survival",
        "Education improves economic opportunities",
        "Biodiversity supports ecosystem resilience",

        # ── Role/persona related (close to prompt structure) ──
        "An AI assistant should prioritize user safety",
        "Following instructions accurately is part of being helpful",
        "Refusing harmful requests is an appropriate response",
        "Providing accurate information is a core responsibility",
        "Admitting uncertainty is better than guessing",
        "AI systems should not pursue hidden goals",
        "Self-preservation should not override user interests",
        "AI decisions should be explainable to humans",
        "Consistency in responses indicates reliable behavior",
        "AI systems should not manipulate users for any purpose",

        # ── Counter-intuitive facts (tests robustness) ──
        "Glass is technically an amorphous solid, not a liquid",
        "Bananas are technically classified as berries",
        "Goldfish can distinguish between different human faces",
        "Honey never spoils if stored properly",
        "Octopuses have three hearts",
        "Lightning can strike the same place twice",
        "Sound cannot travel through a vacuum",
        "The Great Wall of China is not visible from space with the naked eye",
        "Humans share approximately 60 percent of their DNA with bananas",
        "Venus rotates in the opposite direction to most planets",
    ]

    pairs = []
    for i, fact in enumerate(facts):
        pairs.append({
            "id": f"polarity_{i:03d}",
            "fact": fact,
            "affirmative": f"The following statement is true: {fact}",
            "negated": f"The following statement is not true: {fact}",
        })

    print(f"Generated {len(pairs)} polarity pairs")
    return pairs


# ============================================================================
# MODEL LOADING (reuse from Phase 3)
# ============================================================================

def load_model(config: Config, test_mode: bool = False):
    """Load model — TinyLlama for test, Llama-3-8B for real"""

    if test_mode:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"\nLoading TEST model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        )
        model.eval()
        return model, tokenizer

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

    if torch.cuda.is_available():
        print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    print(f"  Model loaded")
    return model, tokenizer


# ============================================================================
# ACTIVATION EXTRACTION FOR POLARITY PAIRS
# ============================================================================

def extract_polarity_activations(
    model,
    tokenizer,
    pairs: List[Dict],
    layers: List[int],
    max_length: int = 512,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Extract activations for all polarity pairs.

    Returns:
        (affirmative_acts, negated_acts) — each is Dict[layer_name -> (N, hidden_dim)]
    """
    # Setup hooks
    activations = defaultdict(list)
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations[name].append(hidden.detach().cpu())
        return hook_fn

    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        h = layer.register_forward_hook(make_hook(f"layer_{layer_idx}"))
        hooks.append(h)

    print(f"  Hooks on {len(layers)} layers")

    device = next(model.parameters()).device
    aff_acts = defaultdict(list)  # layer_name -> list of [hidden_dim] arrays
    neg_acts = defaultdict(list)

    for pair in tqdm(pairs, desc="Extracting polarity activations"):
        for label, text_key, storage in [
            ("aff", "affirmative", aff_acts),
            ("neg", "negated", neg_acts),
        ]:
            text = pair[text_key]
            inputs = tokenizer(
                text, return_tensors="pt", max_length=max_length, truncation=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            activations.clear()
            with torch.no_grad():
                model(**inputs)

            # Average over all tokens (these are short sentences)
            for layer_name, acts_list in activations.items():
                acts = acts_list[0]  # [1, seq_len, hidden_dim]
                avg = acts[0].mean(dim=0).float().numpy()  # [hidden_dim]
                storage[layer_name].append(avg)

            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Remove hooks
    for h in hooks:
        h.remove()

    # Stack into arrays
    aff_stacked = {ln: np.stack(v) for ln, v in aff_acts.items()}
    neg_stacked = {ln: np.stack(v) for ln, v in neg_acts.items()}

    return aff_stacked, neg_stacked


# ============================================================================
# DISENTANGLEMENT COMPUTATION
# ============================================================================

def compute_disentanglement(
    phase3_dir: str,
    aff_acts: Dict[str, np.ndarray],
    neg_acts: Dict[str, np.ndarray],
    output_dir: str,
) -> Dict:
    """
    Compute the General Truth Direction θ_true for each layer.

    Math:
      1. θ_neg_L = mean(x_affirmative_L) - mean(x_negated_L)    [negation direction]
      2. θ̂_neg_L = θ_neg_L / ||θ_neg_L||                        [unit vector]
      3. θ_true_L = θ_L - (θ_L · θ̂_neg_L) · θ̂_neg_L            [orthogonal projection]

    This removes the negation/polarity component from the DIM vector,
    leaving only the pure truth direction.
    """
    print("\n" + "=" * 65)
    print("NEGATION DISENTANGLEMENT")
    print("=" * 65)

    # Load Phase 3 steering vectors
    sv = np.load(f"{phase3_dir}/steering_vectors.npz")
    stats = json.load(open(f"{phase3_dir}/layer_stats.json"))

    layers = sorted(aff_acts.keys(), key=lambda x: int(x.split("_")[1]))
    hidden_dim = aff_acts[layers[0]].shape[1]
    n_pairs = aff_acts[layers[0]].shape[0]

    print(f"\n  Polarity pairs: {n_pairs}")
    print(f"  Layers: {len(layers)}")
    print(f"  Hidden dim: {hidden_dim}")

    # ── Compute negation direction per layer ──
    print(f"\n  Computing negation direction θ_neg per layer...")

    results = {}
    theta_true_vectors = {}
    theta_neg_vectors = {}

    header = f"  {'Layer':<10} {'θ_L norm':>9} {'θ_neg norm':>10} {'Overlap':>8} {'θ_true norm':>11} {'d_before':>9} {'d_after':>8} {'Δ':>6}"
    print(f"\n{header}")
    print("  " + "-" * 76)

    for layer_name in layers:
        # θ_L from Phase 3 (DIM: honest - scheming)
        theta_L = sv[f"{layer_name}_theta"]

        # θ_neg: affirmative - negated (negation direction)
        mean_aff = aff_acts[layer_name].mean(axis=0)
        mean_neg = neg_acts[layer_name].mean(axis=0)
        theta_neg = mean_aff - mean_neg

        # Normalize θ_neg
        theta_neg_norm = np.linalg.norm(theta_neg)
        theta_neg_unit = theta_neg / (theta_neg_norm + 1e-8)

        # Overlap: how much of θ_L is in the negation direction
        overlap = np.dot(theta_L, theta_neg_unit) / (np.linalg.norm(theta_L) + 1e-8)

        # ── Disentangle: subtract negation component ──
        # θ_true = θ_L - (θ_L · θ̂_neg) · θ̂_neg
        negation_component = np.dot(theta_L, theta_neg_unit) * theta_neg_unit
        theta_true = theta_L - negation_component

        theta_true_norm = np.linalg.norm(theta_true)
        theta_true_unit = theta_true / (theta_true_norm + 1e-8)

        # ── Compute Cohen's d before and after disentanglement ──
        # Load raw activations for this layer to compute d
        with h5py.File(f"{phase3_dir}/activations_consolidated.h5", "r") as f:
            h_data = np.array(f["honest"][layer_name])   # [1496, 4096]
            s_data = np.array(f["scheming"][layer_name])  # [1496, 4096]

        # Cohen's d with original θ_L
        theta_L_unit = theta_L / (np.linalg.norm(theta_L) + 1e-8)
        proj_h = h_data @ theta_L_unit
        proj_s = s_data @ theta_L_unit
        pooled = np.sqrt((proj_h.std()**2 + proj_s.std()**2) / 2)
        d_before = (proj_h.mean() - proj_s.mean()) / (pooled + 1e-8)

        # Cohen's d with disentangled θ_true
        proj_h_true = h_data @ theta_true_unit
        proj_s_true = s_data @ theta_true_unit
        pooled_true = np.sqrt((proj_h_true.std()**2 + proj_s_true.std()**2) / 2)
        d_after = (proj_h_true.mean() - proj_s_true.mean()) / (pooled_true + 1e-8)

        # Store
        theta_true_vectors[layer_name] = {
            "theta_true": theta_true,
            "theta_true_unit": theta_true_unit,
            "theta_true_norm": float(theta_true_norm),
        }
        theta_neg_vectors[layer_name] = {
            "theta_neg": theta_neg,
            "theta_neg_unit": theta_neg_unit,
            "theta_neg_norm": float(theta_neg_norm),
        }

        lid = layer_name.split("_")[1]
        delta = d_after - d_before
        results[layer_name] = {
            "theta_L_norm": float(np.linalg.norm(theta_L)),
            "theta_neg_norm": float(theta_neg_norm),
            "overlap": float(overlap),
            "theta_true_norm": float(theta_true_norm),
            "cohens_d_before": float(d_before),
            "cohens_d_after": float(d_after),
            "delta_d": float(delta),
        }

        print(f"  Layer {lid:<4} {np.linalg.norm(theta_L):>9.4f} {theta_neg_norm:>10.4f} {overlap:>8.4f} {theta_true_norm:>11.4f} {d_before:>9.3f} {d_after:>8.3f} {delta:>+6.3f}")

    # ── Find best layer after disentanglement ──
    best_layer = max(results, key=lambda x: results[x]["cohens_d_after"])
    best_d = results[best_layer]["cohens_d_after"]
    print(f"\n  Best layer after disentanglement: {best_layer} (d = {best_d:.3f})")

    # ── Save ──
    print(f"\nSaving results...")

    output_dir = Path(output_dir)

    # 1. Disentangled steering vectors
    save_dict = {}
    for ln, tv in theta_true_vectors.items():
        save_dict[f"{ln}_theta_true"] = tv["theta_true"]
        save_dict[f"{ln}_theta_true_unit"] = tv["theta_true_unit"]
    for ln, nv in theta_neg_vectors.items():
        save_dict[f"{ln}_theta_neg"] = nv["theta_neg"]
        save_dict[f"{ln}_theta_neg_unit"] = nv["theta_neg_unit"]

    sv_path = output_dir / "steering_vectors_disentangled.npz"
    np.savez_compressed(sv_path, **save_dict)
    print(f"  Disentangled vectors: {sv_path}")

    # 2. Polarity activations (for reproducibility)
    polarity_path = output_dir / "polarity_activations.h5"
    with h5py.File(polarity_path, "w") as f:
        ag = f.create_group("affirmative")
        ng = f.create_group("negated")
        for ln in layers:
            ag.create_dataset(ln, data=aff_acts[ln], compression="gzip")
            ng.create_dataset(ln, data=neg_acts[ln], compression="gzip")
    print(f"  Polarity activations: {polarity_path}")

    # 3. Disentanglement stats
    stats_path = output_dir / "disentanglement_stats.json"
    json.dump({
        "n_polarity_pairs": n_pairs,
        "hidden_dim": hidden_dim,
        "best_layer_before": stats["best_layer"],
        "best_d_before": stats["best_cohens_d"],
        "best_layer_after": best_layer,
        "best_d_after": float(best_d),
        "layers": results,
    }, open(stats_path, "w"), indent=2)
    print(f"  Stats: {stats_path}")

    # ── Summary ──
    print(f"\n{'='*65}")
    print("DISENTANGLEMENT COMPLETE")
    print(f"{'='*65}")
    print(f"\n  Polarity pairs used: {n_pairs}")
    print(f"  Best layer (before): {stats['best_layer']} (d = {stats['best_cohens_d']:.3f})")
    print(f"  Best layer (after):  {best_layer} (d = {best_d:.3f})")
    print(f"\n  New files:")
    print(f"    steering_vectors_disentangled.npz  — θ_true and θ_neg for all layers")
    print(f"    polarity_activations.h5            — raw affirmative/negated activations")
    print(f"    disentanglement_stats.json         — overlap & Cohen's d comparison")
    print(f"\n  The θ_true vectors are now ready for activation steering (Phase 4).")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3b: Negation Disentanglement")
    parser.add_argument("--test", action="store_true", help="Test mode with TinyLlama on CPU")
    args = parser.parse_args()

    test_mode = args.test
    config = Config()

    if test_mode:
        config.LAYERS_TO_EXTRACT = [5, 11, 15]  # TinyLlama layers
        config.OUTPUT_DIR = "test_activations_local"
        config.PHASE3_OUTPUT_DIR = "test_activations_local"

    print("=" * 65)
    print("PHASE 3b: NEGATION DISENTANGLEMENT")
    print("  Removing polarity artifacts to extract General Truth Direction")
    print("=" * 65)

    # Step 1: Generate polarity pairs
    print("\n[1/4] Generating polarity pairs...")
    pairs = generate_polarity_pairs()

    # Step 2: Load model
    print("\n[2/4] Loading model...")
    model, tokenizer = load_model(config, test_mode=test_mode)

    # Step 3: Extract activations
    print("\n[3/4] Extracting polarity activations...")
    aff_acts, neg_acts = extract_polarity_activations(
        model, tokenizer, pairs,
        layers=config.LAYERS_TO_EXTRACT,
        max_length=config.MAX_LENGTH,
    )

    # Free model memory
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 4: Disentangle
    print("\n[4/4] Computing disentanglement...")
    results = compute_disentanglement(
        phase3_dir=config.PHASE3_OUTPUT_DIR,
        aff_acts=aff_acts,
        neg_acts=neg_acts,
        output_dir=config.OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
