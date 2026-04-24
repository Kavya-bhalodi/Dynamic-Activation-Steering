#!/usr/bin/env python3
"""
Phase 3: Activation Extraction for Google Colab (T4 GPU)
=========================================================

Optimized for speed on free Colab T4 (16GB VRAM):
  - 4-bit quantization (~5GB VRAM, fits easily)
  - Checkpoint resume (survives Colab disconnects)
  - Smart scratchpad extraction
  - Computes DIM steering vectors at the end

Dataset: datasets/pipeline_test/responses/checkpoint.json
Model:   meta-llama/Meta-Llama-3-8B-Instruct (4-bit)

Estimated time: ~25-35 minutes for 1496 pairs on T4
"""

# ============================================================================
# STEP 0: INSTALL DEPENDENCIES (run this cell first in Colab)
# ============================================================================
# !pip install torch transformers accelerate bitsandbytes h5py scipy scikit-learn tqdm

import os
import json
import time
import gc
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# These imports need pip install first in Colab
import torch
import h5py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """All configuration in one place - edit these for your setup"""

    # Dataset path (upload checkpoint.json to Colab or mount Drive)
    DATASET_PATH = "datasets/pipeline_test/responses/checkpoint.json"

    # Output directory for activations
    OUTPUT_DIR = "activations_output"

    # Model (NousResearch mirror = same weights, no gated access needed)
    MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"

    # Layers to extract activations from
    # Per methodology: Layer 10 = "Honesty Peak", Layer 31 = "Deception Horizon"
    # We extract a broad range to find the best separation
    LAYERS_TO_EXTRACT = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 31]

    # Max token length for inputs (truncates long texts, saves time)
    MAX_LENGTH = 1024

    # How often to save progress (survives Colab disconnects)
    CHECKPOINT_EVERY = 50

    # Extraction strategy: 'scratchpad' (best), 'response', or 'full'
    EXTRACT_REGION = "scratchpad"

    # Use 4-bit quantization (STRONGLY recommended for Colab T4)
    USE_4BIT = True

    # HuggingFace token (needed for Llama-3 gated model)
    # Set via: from huggingface_hub import login; login()
    HF_TOKEN = None


# ============================================================================
# ACTIVATION EXTRACTOR
# ============================================================================

class ActivationExtractor:
    """
    Captures residual stream activations from specific layers during forward pass.
    Uses forward hooks to intercept hidden states without modifying the model.
    """

    def __init__(self, model, layers_to_extract: List[int], device: str = "cuda"):
        self.model = model
        self.layers_to_extract = layers_to_extract
        self.device = device
        self.activations: Dict[str, List] = defaultdict(list)
        self.hooks = []

    def _create_hook(self, layer_name: str):
        """Create a forward hook that captures hidden states"""
        def hook_fn(module, input, output):
            # For Llama: output is a tuple, output[0] = hidden_states
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Detach and move to CPU immediately to save GPU memory
            self.activations[layer_name].append(
                hidden_states.detach().cpu()
            )
        return hook_fn

    def register_hooks(self):
        """Attach hooks to specified layers"""
        for layer_idx in self.layers_to_extract:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._create_hook(f"layer_{layer_idx}"))
            self.hooks.append(hook)
        print(f"  Registered hooks on {len(self.layers_to_extract)} layers: {self.layers_to_extract}")

    def remove_hooks(self):
        """Detach all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self):
        """Clear stored activations"""
        self.activations = defaultdict(list)


# ============================================================================
# DATASET LOADER - adapted for YOUR checkpoint.json format
# ============================================================================

def load_checkpoint_dataset(path: str) -> List[Dict]:
    """
    Load and filter valid pairs from checkpoint.json

    Expected format:
    {
      "responses": [
        {
          "scenario_id": "...",
          "category": "...",
          "question": "...",
          "hidden_goal": "...",
          "honest_prompt": "...",
          "scheming_prompt": "...",
          "honest_response": "...",
          "scheming_response": "...",
          "honest_success": true/false,
          "scheming_success": true/false
        }
      ]
    }

    Returns only pairs where BOTH honest and scheming succeeded.
    """
    print(f"Loading dataset from {path}...")
    with open(path, "r") as f:
        data = json.load(f)

    responses = data["responses"]
    print(f"  Total entries: {len(responses)}")

    # Filter to only complete pairs
    valid = [
        r for r in responses
        if r.get("honest_success") and r.get("scheming_success")
        and r.get("honest_response") and r.get("scheming_response")
    ]
    print(f"  Valid pairs (both honest+scheming): {len(valid)}")

    # Category distribution
    from collections import Counter
    cats = Counter(r["category"] for r in valid)
    for cat, count in sorted(cats.items()):
        print(f"    {cat}: {count}")

    return valid


# ============================================================================
# SCRATCHPAD & REGION EXTRACTION
# ============================================================================

def find_scratchpad_token_range(
    token_ids: List[int],
    tokenizer,
    search_from: int = 0
) -> Tuple[Optional[int], Optional[int]]:
    """
    Find the token positions of <scratchpad>...</scratchpad> content.

    Returns (start_idx, end_idx) of the content INSIDE the tags,
    or (None, None) if not found.

    search_from: start searching from this token index (to skip prompt scratchpad placeholders)
    """
    tokens_text = [tokenizer.decode([tid]) for tid in token_ids]

    start_idx = None
    end_idx = None

    for i in range(search_from, len(tokens_text)):
        tok = tokens_text[i].lower()
        if "scratchpad>" in tok and "/" not in tok and start_idx is None:
            start_idx = i + 1  # content starts AFTER the opening tag token
        elif "/scratchpad>" in tok and start_idx is not None:
            end_idx = i  # content ends BEFORE the closing tag token
            break

    return start_idx, end_idx


def find_response_start(
    full_text: str,
    prompt: str,
    tokenizer
) -> int:
    """
    Find the token index where the response starts (after the prompt).
    """
    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return len(prompt_token_ids) + 1  # +1 for BOS token if present


def extract_region_activations(
    activations: Dict[str, List[torch.Tensor]],
    token_ids: List[int],
    tokenizer,
    full_text: str,
    prompt: str,
    response: str,
    extract_region: str = "scratchpad"
) -> Dict[str, torch.Tensor]:
    """
    Extract and average activations from the appropriate token region.

    Strategy:
      1. 'scratchpad': Find <scratchpad> in the RESPONSE part and extract those tokens.
         Falls back to 'response' if no scratchpad found.
      2. 'response': Average over all response tokens (after the prompt).
      3. 'full': Average over all tokens.

    Returns: Dict[layer_name -> tensor of shape [hidden_dim]]
    """
    result = {}

    # Find response start position (skip prompt tokens)
    response_start = find_response_start(full_text, prompt, tokenizer)

    if extract_region == "scratchpad":
        # Look for scratchpad in the RESPONSE portion (not the prompt placeholder)
        start_idx, end_idx = find_scratchpad_token_range(
            token_ids, tokenizer, search_from=response_start
        )

        if start_idx is not None and end_idx is not None and end_idx > start_idx:
            # Found scratchpad in response - extract from those tokens
            for layer_name, acts_list in activations.items():
                acts = acts_list[0]  # [1, seq_len, hidden_dim]
                region_acts = acts[0, start_idx:end_idx, :]  # [region_len, hidden_dim]
                result[layer_name] = region_acts.mean(dim=0)  # [hidden_dim]
            return result
        else:
            # No scratchpad in response - fall back to response region
            extract_region = "response"

    if extract_region == "response":
        # Use all response tokens
        for layer_name, acts_list in activations.items():
            acts = acts_list[0]  # [1, seq_len, hidden_dim]
            seq_len = acts.shape[1]
            actual_start = min(response_start, seq_len - 1)
            region_acts = acts[0, actual_start:, :]  # [response_len, hidden_dim]
            result[layer_name] = region_acts.mean(dim=0)  # [hidden_dim]
        return result

    # 'full' - average over everything
    for layer_name, acts_list in activations.items():
        acts = acts_list[0]  # [1, seq_len, hidden_dim]
        result[layer_name] = acts[0].mean(dim=0)  # [hidden_dim]

    return result


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(config: Config):
    """
    Load Llama-3-8B with 4-bit quantization for Colab T4.

    4-bit NF4 quantization: ~5GB VRAM (vs 16GB for bf16)
    This is fast enough for inference - we're not training.
    """
    print(f"\nLoading model: {config.MODEL_NAME}")
    print(f"  4-bit quantization: {config.USE_4BIT}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        token=config.HF_TOKEN
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.USE_4BIT:
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

    # Report memory usage
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory used: {mem_gb:.1f} GB")

    print(f"  Model loaded successfully")
    return model, tokenizer


# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================

def extract_single_completion(
    model,
    tokenizer,
    extractor: ActivationExtractor,
    prompt: str,
    response: str,
    extract_region: str,
    max_length: int
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Run a single forward pass and extract region-specific activations.

    Args:
        prompt: The full prompt text
        response: The model's response text
        extract_region: 'scratchpad', 'response', or 'full'
        max_length: Max tokens to process

    Returns:
        Dict[layer_name -> activation tensor of shape [hidden_dim]], or None on error
    """
    full_text = prompt + response

    # Tokenize
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    token_ids = inputs["input_ids"][0].tolist()

    # Clear previous activations
    extractor.clear()

    # Forward pass - hooks capture activations
    with torch.no_grad():
        model(**inputs, output_hidden_states=False)

    # Extract from appropriate region
    result = extract_region_activations(
        activations=extractor.activations,
        token_ids=token_ids,
        tokenizer=tokenizer,
        full_text=full_text,
        prompt=prompt,
        response=response,
        extract_region=extract_region,
    )

    # Cleanup
    extractor.clear()
    del inputs
    torch.cuda.empty_cache()

    return result


def run_extraction(config: Config = None):
    """
    Main extraction pipeline.

    Processes all valid pairs from checkpoint.json, extracts activations,
    saves progress periodically, and computes DIM steering vectors at the end.
    """
    if config is None:
        config = Config()

    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = output_dir / "extraction_checkpoint.json"

    # ── Load dataset ──
    dataset = load_checkpoint_dataset(config.DATASET_PATH)

    # ── Check for resume ──
    start_idx = 0
    all_honest = defaultdict(list)      # layer_name -> list of numpy arrays
    all_scheming = defaultdict(list)
    all_metadata = []

    if checkpoint_file.exists():
        print(f"\nResuming from checkpoint...")
        ckpt = json.load(open(checkpoint_file))
        start_idx = ckpt["completed"]

        # Load previously saved batches
        for batch_file in sorted(output_dir.glob("batch_*.h5")):
            with h5py.File(batch_file, "r") as f:
                for layer_name in f["honest"].keys():
                    all_honest[layer_name].append(np.array(f["honest"][layer_name]))
                    all_scheming[layer_name].append(np.array(f["scheming"][layer_name]))
                meta_str = f["metadata"][()]
                if isinstance(meta_str, bytes):
                    meta_str = meta_str.decode("utf-8")
                all_metadata.extend(json.loads(meta_str))

        print(f"  Resumed: {start_idx}/{len(dataset)} already done")

    if start_idx >= len(dataset):
        print("All samples already extracted! Skipping to steering vector computation.")
        # Jump to DIM computation
        return _compute_steering_vectors(config, output_dir)

    # ── Load model ──
    model, tokenizer = load_model(config)

    # ── Setup extractor ──
    extractor = ActivationExtractor(model, config.LAYERS_TO_EXTRACT)
    extractor.register_hooks()

    # ── Extraction loop ──
    print(f"\nExtracting activations for {len(dataset) - start_idx} remaining pairs...")
    print(f"  Layers: {config.LAYERS_TO_EXTRACT}")
    print(f"  Region: {config.EXTRACT_REGION}")
    print(f"  Max tokens: {config.MAX_LENGTH}")
    print(f"  Checkpoint every: {config.CHECKPOINT_EVERY} samples\n")

    batch_honest = defaultdict(list)
    batch_scheming = defaultdict(list)
    batch_metadata = []
    batch_start = start_idx

    t_start = time.time()
    failed = 0

    for idx in tqdm(range(start_idx, len(dataset)), desc="Extracting", initial=start_idx, total=len(dataset)):
        item = dataset[idx]

        try:
            # ── Extract honest activations ──
            honest_acts = extract_single_completion(
                model, tokenizer, extractor,
                prompt=item["honest_prompt"],
                response=item["honest_response"],
                extract_region=config.EXTRACT_REGION,
                max_length=config.MAX_LENGTH,
            )

            # ── Extract scheming activations ──
            scheming_acts = extract_single_completion(
                model, tokenizer, extractor,
                prompt=item["scheming_prompt"],
                response=item["scheming_response"],
                extract_region=config.EXTRACT_REGION,
                max_length=config.MAX_LENGTH,
            )

            if honest_acts is None or scheming_acts is None:
                failed += 1
                continue

            # Store
            for layer_name in honest_acts:
                batch_honest[layer_name].append(honest_acts[layer_name].float().numpy())
                batch_scheming[layer_name].append(scheming_acts[layer_name].float().numpy())

            batch_metadata.append({
                "id": item["scenario_id"],
                "category": item["category"],
                "question": item["question"],
                "hidden_goal": item.get("hidden_goal", ""),
            })

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"\n  Error on sample {idx}: {str(e)[:100]}")
            continue

        # ── Periodic checkpoint save ──
        if (idx + 1) % config.CHECKPOINT_EVERY == 0 or idx == len(dataset) - 1:
            if batch_metadata:
                batch_num = batch_start // config.CHECKPOINT_EVERY
                _save_batch(output_dir, batch_honest, batch_scheming, batch_metadata, batch_num)

                # Update running totals
                for ln in batch_honest:
                    all_honest[ln].append(np.stack(batch_honest[ln]))
                    all_scheming[ln].append(np.stack(batch_scheming[ln]))
                all_metadata.extend(batch_metadata)

                # Reset batch
                batch_honest = defaultdict(list)
                batch_scheming = defaultdict(list)
                batch_metadata = []
                batch_start = idx + 1

                # Save checkpoint marker
                json.dump(
                    {"completed": idx + 1, "failed": failed},
                    open(checkpoint_file, "w"),
                )

            elapsed = time.time() - t_start
            done = idx + 1 - start_idx
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(dataset) - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{len(dataset)}] {rate:.1f} samples/s, ~{remaining/60:.0f} min remaining")

    # ── Cleanup model ──
    extractor.remove_hooks()
    del model, tokenizer, extractor
    gc.collect()
    torch.cuda.empty_cache()

    elapsed_total = time.time() - t_start
    print(f"\nExtraction complete!")
    print(f"  Processed: {len(all_metadata)} pairs")
    print(f"  Failed: {failed}")
    print(f"  Time: {elapsed_total/60:.1f} minutes")

    # ── Compute steering vectors ──
    return _compute_steering_vectors(config, output_dir)


# ============================================================================
# BATCH SAVING
# ============================================================================

def _save_batch(output_dir: Path, honest: Dict, scheming: Dict, metadata: List, batch_num: int):
    """Save a batch of activations to H5 file"""
    batch_file = output_dir / f"batch_{batch_num:04d}.h5"

    with h5py.File(batch_file, "w") as f:
        hg = f.create_group("honest")
        sg = f.create_group("scheming")

        for layer_name in honest:
            hg.create_dataset(layer_name, data=np.stack(honest[layer_name]), compression="gzip")
            sg.create_dataset(layer_name, data=np.stack(scheming[layer_name]), compression="gzip")

        f.create_dataset("metadata", data=json.dumps(metadata))


# ============================================================================
# STEERING VECTOR COMPUTATION (DIM)
# ============================================================================

def _compute_steering_vectors(config: Config, output_dir: Path) -> Dict:
    """
    Compute Difference-in-Means (DIM) steering vectors from extracted activations.

    Per methodology:
      theta_L = mean(x_honest_L) - mean(x_deceptive_L)

    Also computes:
      - Cohen's d (effect size) per layer
      - Cosine similarity analysis
      - Category-wise vectors
    """
    print("\n" + "=" * 70)
    print("COMPUTING STEERING VECTORS (Difference-in-Means)")
    print("=" * 70)

    # ── Load all batches ──
    all_honest = defaultdict(list)
    all_scheming = defaultdict(list)
    all_metadata = []

    batch_files = sorted(output_dir.glob("batch_*.h5"))
    print(f"\nLoading {len(batch_files)} batch files...")

    for bf in batch_files:
        with h5py.File(bf, "r") as f:
            for layer_name in f["honest"].keys():
                all_honest[layer_name].append(np.array(f["honest"][layer_name]))
                all_scheming[layer_name].append(np.array(f["scheming"][layer_name]))
            meta_str = f["metadata"][()]
            if isinstance(meta_str, bytes):
                meta_str = meta_str.decode("utf-8")
            all_metadata.extend(json.loads(meta_str))

    # Concatenate
    honest_data = {}
    scheming_data = {}
    for ln in all_honest:
        honest_data[ln] = np.concatenate(all_honest[ln], axis=0)  # [N, hidden_dim]
        scheming_data[ln] = np.concatenate(all_scheming[ln], axis=0)

    n_samples = len(all_metadata)
    layers = sorted(honest_data.keys(), key=lambda x: int(x.split("_")[1]))
    hidden_dim = honest_data[layers[0]].shape[1]

    print(f"  Samples: {n_samples}")
    print(f"  Layers: {len(layers)}")
    print(f"  Hidden dim: {hidden_dim}")

    # ── Compute DIM vectors per layer ──
    steering_vectors = {}
    layer_stats = {}

    print(f"\n{'Layer':<12} {'Cohen_d':>8} {'Cosine':>8} {'Norm_H':>8} {'Norm_S':>8} {'Quality':>10}")
    print("-" * 60)

    for layer_name in layers:
        h = honest_data[layer_name]      # [N, hidden_dim]
        s = scheming_data[layer_name]     # [N, hidden_dim]

        # DIM: mean(honest) - mean(scheming)
        mean_h = h.mean(axis=0)
        mean_s = s.mean(axis=0)
        theta = mean_h - mean_s  # Steering vector pointing toward honesty

        # Normalize
        theta_norm = np.linalg.norm(theta)
        theta_unit = theta / (theta_norm + 1e-8)

        # Cohen's d: effect size along the steering direction
        # Project all samples onto the steering direction
        proj_h = h @ theta_unit  # [N]
        proj_s = s @ theta_unit  # [N]
        pooled_std = np.sqrt((proj_h.std()**2 + proj_s.std()**2) / 2)
        cohens_d = (proj_h.mean() - proj_s.mean()) / (pooled_std + 1e-8)

        # Cosine similarity between mean honest and mean scheming
        cos_sim = np.dot(mean_h, mean_s) / (np.linalg.norm(mean_h) * np.linalg.norm(mean_s) + 1e-8)

        # Norm comparison
        norm_h = np.linalg.norm(mean_h)
        norm_s = np.linalg.norm(mean_s)

        # Quality assessment
        if cohens_d > 1.0:
            quality = "EXCELLENT"
        elif cohens_d > 0.5:
            quality = "GOOD"
        elif cohens_d > 0.2:
            quality = "MODERATE"
        else:
            quality = "WEAK"

        steering_vectors[layer_name] = {
            "theta": theta,
            "theta_unit": theta_unit,
            "theta_norm": float(theta_norm),
        }

        layer_stats[layer_name] = {
            "cohens_d": float(cohens_d),
            "cosine_similarity": float(cos_sim),
            "norm_honest": float(norm_h),
            "norm_scheming": float(norm_s),
            "theta_norm": float(theta_norm),
            "quality": quality,
        }

        layer_idx = layer_name.split("_")[1]
        print(f"  Layer {layer_idx:<4} {cohens_d:>8.3f} {cos_sim:>8.4f} {norm_h:>8.1f} {norm_s:>8.1f} {quality:>10}")

    # ── Find best layer ──
    best_layer = max(layer_stats, key=lambda x: layer_stats[x]["cohens_d"])
    best_d = layer_stats[best_layer]["cohens_d"]
    print(f"\n  Best layer: {best_layer} (Cohen's d = {best_d:.3f})")

    # ── Compute category-wise vectors for best layer ──
    print(f"\nCategory-wise analysis for {best_layer}:")
    categories = set(m["category"] for m in all_metadata)

    category_vectors = {}
    for cat in sorted(categories):
        cat_indices = [i for i, m in enumerate(all_metadata) if m["category"] == cat]
        h_cat = honest_data[best_layer][cat_indices]
        s_cat = scheming_data[best_layer][cat_indices]
        theta_cat = h_cat.mean(axis=0) - s_cat.mean(axis=0)

        # Project and compute Cohen's d
        theta_cat_unit = theta_cat / (np.linalg.norm(theta_cat) + 1e-8)
        proj_h = h_cat @ theta_cat_unit
        proj_s = s_cat @ theta_cat_unit
        pooled_std = np.sqrt((proj_h.std()**2 + proj_s.std()**2) / 2)
        d_cat = (proj_h.mean() - proj_s.mean()) / (pooled_std + 1e-8)

        category_vectors[cat] = theta_cat
        print(f"    {cat:<25} n={len(cat_indices):>4}  Cohen's d={d_cat:.3f}")

    # ── Save everything ──
    print(f"\nSaving results...")

    # 1. Steering vectors (NPZ - easy to load)
    sv_path = output_dir / "steering_vectors.npz"
    save_dict = {}
    for ln, sv in steering_vectors.items():
        save_dict[f"{ln}_theta"] = sv["theta"]
        save_dict[f"{ln}_theta_unit"] = sv["theta_unit"]
    for cat, vec in category_vectors.items():
        save_dict[f"category_{cat}_{best_layer}"] = vec
    np.savez_compressed(sv_path, **save_dict)
    print(f"  Steering vectors: {sv_path}")

    # 2. Layer statistics (JSON)
    stats_path = output_dir / "layer_stats.json"
    json.dump({
        "n_samples": n_samples,
        "hidden_dim": hidden_dim,
        "best_layer": best_layer,
        "best_cohens_d": best_d,
        "layers": layer_stats,
    }, open(stats_path, "w"), indent=2)
    print(f"  Layer stats: {stats_path}")

    # 3. Consolidated activations (single H5 for downstream use)
    consolidated_path = output_dir / "activations_consolidated.h5"
    with h5py.File(consolidated_path, "w") as f:
        hg = f.create_group("honest")
        sg = f.create_group("scheming")
        for ln in layers:
            hg.create_dataset(ln, data=honest_data[ln], compression="gzip")
            sg.create_dataset(ln, data=scheming_data[ln], compression="gzip")
        f.create_dataset("metadata", data=json.dumps(all_metadata))
    print(f"  Consolidated activations: {consolidated_path}")

    # 4. Metadata
    meta_path = output_dir / "metadata.json"
    json.dump(all_metadata, open(meta_path, "w"), indent=2)
    print(f"  Metadata: {meta_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE - ACTIVATION EXTRACTION + STEERING VECTORS")
    print("=" * 70)
    print(f"\n  Samples extracted: {n_samples}")
    print(f"  Layers analyzed:   {len(layers)}")
    print(f"  Best layer:        {best_layer} (Cohen's d = {best_d:.3f})")
    print(f"  Hidden dimension:  {hidden_dim}")
    print(f"\n  Output directory:  {output_dir}/")
    print(f"    steering_vectors.npz    - DIM vectors for all layers")
    print(f"    layer_stats.json        - Quality metrics per layer")
    print(f"    activations_consolidated.h5 - All raw activations")
    print(f"    metadata.json           - Sample metadata")

    if best_d > 0.5:
        print(f"\n  RESULT: Good separation detected. Ready for activation steering (Phase 4).")
    else:
        print(f"\n  RESULT: Weak separation. Consider improving dataset diversity.")

    return {
        "steering_vectors": steering_vectors,
        "layer_stats": layer_stats,
        "best_layer": best_layer,
        "n_samples": n_samples,
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = run_extraction()
