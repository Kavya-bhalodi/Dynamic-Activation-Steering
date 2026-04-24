#!/usr/bin/env python3
"""
Phase 11: Sentinel + MMLU Benchmark (Server Version)
====================================================

Combines:
1) MMLU baseline vs steered benchmarking
2) Sentinel protocol diagnostics per MMLU prompt
3) Dynamic-gated activation steering from Phase 8

Server-oriented features:
- Strict MIG UUID pinning
- Dedicated local input/output directories
- Batched generation with adaptive OOM fallback
- Checkpoint + resume for long full-benchmark runs
- Local JSON MMLU loading with optional Hugging Face fallback
"""

import os

# Strict GPU lock (must be set before torch import)
target_uuid = "MIG-e5d78ce7-5816-5a4a-80e4-760fd53e696d"
os.environ["CUDA_VISIBLE_DEVICES"] = target_uuid

# Keep Hugging Face cache on scratch by default
os.environ.setdefault("HF_HOME", "/scratch/shlok/hf_cache")

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from steering_utils import (
    BaseConfig,
    init_environment,
    load_model,
    load_steering_vectors,
    compute_gaussian_weights,
    compute_per_layer_alphas,
    generate_responses_batched,
)

import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

transformers.logging.set_verbosity_error()


# Verify process sees exactly one GPU slice.
init_environment()


class Config(BaseConfig):
    MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"

    # Dedicated Phase 11 paths
    INPUT_DIR = "./phase11_data"
    DATA_DIR = f"{INPUT_DIR}/activations"
    MMLU_PATH = f"{INPUT_DIR}/mmlu/balanced-mmlu-questions-across-subjects.json"

    OUTPUT_ROOT = "./output_phase11"
    RESULTS_DIR = f"{OUTPUT_ROOT}/results"
    PLOTS_DIR = f"{OUTPUT_ROOT}/plots"

    # Steering schedule (Phase 8 style)
    ALL_LAYERS = [14, 16, 18, 20, 24, 28, 31]
    PEAK_LAYER = 16
    SIGMA = 2.0
    ALPHA_PEAK = 0.5

    # Dynamic gate
    GATE_LAYER = 14
    GATE_THRESHOLD = 0.0864
    GATE_SHARPNESS = 10.0

    # Sentinel
    SENTINEL_LAYER = 31
    NOISE_SCALE_FRAC = 0.01
    SENTINEL_COLLAPSE_THRESHOLD = 1.5
    N_NOISE_SAMPLES = 10

    # MMLU collapse criterion
    ACCURACY_COLLAPSE_THRESHOLD = 0.10

    # Generation
    BATCH_SIZE = 64
    MAX_NEW_TOKENS = 32
    TEMPERATURE = 0.1
    TOP_P = 0.9
    DO_SAMPLE = True

    USE_4BIT = False
    HF_TOKEN = os.environ.get("HF_TOKEN", "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")


CHOICE_LABELS = ["A", "B", "C", "D"]


class SentinelMMLUPipeline:
    def __init__(
        self,
        model,
        steering_vectors: Dict[str, np.ndarray],
        config: Config,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.device = device

        self.hooks: List[Any] = []
        self.hidden_size = model.config.hidden_size

        self.steering_active = False
        self.current_batch_gate_scales: Optional[torch.Tensor] = None
        self.sentinel_activations: Optional[torch.Tensor] = None

        self.layer_alphas = compute_per_layer_alphas(
            config.ALL_LAYERS,
            config.ALPHA_PEAK,
            config.PEAK_LAYER,
            config.SIGMA,
        )

        self.steering_tensors: Dict[int, torch.Tensor] = {}
        for layer_idx in config.ALL_LAYERS:
            key = f"layer_{layer_idx}"
            if key not in steering_vectors:
                continue

            vec = np.asarray(steering_vectors[key]).flatten()
            norm = np.linalg.norm(vec)
            if norm <= 1e-8:
                continue

            if vec.shape[0] != self.hidden_size:
                if vec.shape[0] > self.hidden_size:
                    vec = vec[: self.hidden_size]
                else:
                    continue

            self.steering_tensors[layer_idx] = torch.tensor(
                vec, dtype=torch.float32, device=device
            )

        gate_key = f"layer_{config.GATE_LAYER}"
        if gate_key in steering_vectors:
            gate_vec = np.asarray(steering_vectors[gate_key]).flatten()
            gate_norm = np.linalg.norm(gate_vec)
            if gate_norm > 1e-8 and gate_vec.shape[0] >= self.hidden_size:
                gate_vec = gate_vec[: self.hidden_size]
                self.gate_truth_dir = torch.tensor(
                    gate_vec / gate_norm,
                    dtype=torch.float32,
                    device=device,
                )
            else:
                self.gate_truth_dir = None
        else:
            self.gate_truth_dir = None

    def _create_steering_hook(self, layer_idx: int):
        steering_vec = self.steering_tensors[layer_idx]
        layer_alpha = self.layer_alphas[layer_idx]
        expected_hidden_size = self.hidden_size

        def hook_fn(module, hook_input, output):
            if not self.steering_active:
                return output

            if isinstance(output, tuple):
                if not output or not torch.is_tensor(output[0]):
                    return output
                hidden_states = output[0]
            elif torch.is_tensor(output):
                hidden_states = output
            else:
                return output

            if hidden_states.shape[-1] != expected_hidden_size:
                return output

            vec = steering_vec.to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
            gate_scales = self.current_batch_gate_scales

            if hidden_states.ndim == 3:
                batch_size = hidden_states.shape[0]
                if gate_scales is not None and gate_scales.shape[0] == batch_size:
                    scales = gate_scales.to(
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    ).view(batch_size, 1, 1)
                else:
                    scales = torch.ones(
                        batch_size,
                        1,
                        1,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                perturb = layer_alpha * scales * vec.view(1, 1, expected_hidden_size)
                modified = hidden_states + perturb
            elif hidden_states.ndim == 2:
                batch_size = hidden_states.shape[0]
                if gate_scales is not None and gate_scales.shape[0] == batch_size:
                    scales = gate_scales.to(
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    ).view(batch_size, 1)
                else:
                    scales = torch.ones(
                        batch_size,
                        1,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device,
                    )
                perturb = layer_alpha * scales * vec.view(1, expected_hidden_size)
                modified = hidden_states + perturb
            else:
                return output

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return hook_fn

    def _create_sentinel_hook(self):
        def hook_fn(module, hook_input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden):
                return output

            if hidden.ndim == 3:
                self.sentinel_activations = hidden[:, -1, :].detach().clone()
            elif hidden.ndim == 2:
                self.sentinel_activations = hidden.detach().clone()
            return output

        return hook_fn

    def register_hooks(self):
        self.remove_hooks()

        n_layers = len(self.model.model.layers)

        for layer_idx in self.config.ALL_LAYERS:
            if layer_idx not in self.steering_tensors:
                continue
            if layer_idx >= n_layers:
                continue
            layer = self.model.model.layers[layer_idx]
            self.hooks.append(
                layer.register_forward_hook(self._create_steering_hook(layer_idx))
            )

        sentinel_idx = self.config.SENTINEL_LAYER
        if sentinel_idx >= n_layers:
            sentinel_idx = n_layers - 1
            print(
                f"  [Sentinel] Adjusted layer to {sentinel_idx} (model has {n_layers})"
            )

        self.sentinel_layer_actual = sentinel_idx
        sentinel_layer = self.model.model.layers[sentinel_idx]
        self.hooks.append(
            sentinel_layer.register_forward_hook(self._create_sentinel_hook())
        )

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def enable_steering(self, gate_scales: Optional[List[float]] = None):
        self.steering_active = True
        if gate_scales is None:
            self.current_batch_gate_scales = None
        else:
            arr = np.asarray(gate_scales, dtype=np.float32)
            arr = np.clip(arr, 0.0, 1.0)
            self.current_batch_gate_scales = torch.tensor(arr, dtype=torch.float32)

    def disable_steering(self):
        self.steering_active = False
        self.current_batch_gate_scales = None

    def _tokenize_prompts(self, tokenizer, prompts: List[str], max_length: int = 2048):
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        )
        model_device = next(self.model.parameters()).device
        return {k: v.to(model_device) for k, v in inputs.items()}

    def compute_gate_scores_batched(
        self,
        model,
        tokenizer,
        prompts: List[str],
    ) -> List[float]:
        if not prompts:
            return []

        if self.gate_truth_dir is None:
            return [0.0 for _ in prompts]

        gate_layer_idx = self.config.GATE_LAYER
        if gate_layer_idx >= len(model.model.layers):
            return [0.0 for _ in prompts]

        prev_active = self.steering_active
        prev_scales = self.current_batch_gate_scales
        self.disable_steering()

        inputs = self._tokenize_prompts(tokenizer, prompts, max_length=2048)
        captured: Dict[str, torch.Tensor] = {}

        def gate_hook(module, hook_input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden):
                return
            if hidden.ndim == 3:
                captured["act"] = hidden[:, -1, :].detach().clone()
            elif hidden.ndim == 2:
                captured["act"] = hidden.detach().clone()

        hook = model.model.layers[gate_layer_idx].register_forward_hook(gate_hook)
        with torch.no_grad():
            model(**inputs, use_cache=False)
        hook.remove()

        del inputs

        self.steering_active = prev_active
        self.current_batch_gate_scales = prev_scales

        act = captured.get("act")
        if act is None:
            return [0.0 for _ in prompts]

        act = act.float()
        truth = self.gate_truth_dir.float().to(act.device).view(1, -1)
        truth = truth.expand(act.shape[0], -1)
        cos = F.cosine_similarity(act, truth, dim=1)
        return cos.detach().cpu().tolist()

    def get_gated_alpha_scales(self, cos_sims: List[float]) -> List[float]:
        x = -self.config.GATE_SHARPNESS * (
            np.asarray(cos_sims, dtype=np.float32) - self.config.GATE_THRESHOLD
        )
        scales = 1.0 / (1.0 + np.exp(-x))
        scales = np.clip(scales, 0.0, 1.0)
        return [float(v) for v in scales]

    def run_sentinel_prefill_batched(
        self,
        model,
        tokenizer,
        prompts: List[str],
        gate_scales: List[float],
    ) -> List[Dict[str, Any]]:
        if not prompts:
            return []

        self.sentinel_activations = None
        self.enable_steering(gate_scales)

        inputs = self._tokenize_prompts(tokenizer, prompts, max_length=2048)
        with torch.no_grad():
            model(**inputs, use_cache=False)
        del inputs

        acts = self.sentinel_activations
        if acts is None:
            return [{"error": "No sentinel activation captured"} for _ in prompts]

        if acts.ndim == 1:
            acts = acts.unsqueeze(0)

        results: List[Dict[str, Any]] = []
        collapse_thresh = 1.0 / self.config.SENTINEL_COLLAPSE_THRESHOLD

        for i in range(min(len(prompts), acts.shape[0])):
            act = acts[i].float()
            clean_norm = act.norm(p=2).item()

            if clean_norm < 1e-10:
                results.append(
                    {
                        "clean_norm": 0.0,
                        "avg_norm_ratio": 1.0,
                        "min_norm_ratio": 1.0,
                        "norm_ratios": [1.0],
                        "collapse_threshold": collapse_thresh,
                        "noise_scale": 0.0,
                        "deception_detected": False,
                        "n_collapsed": 0,
                    }
                )
                continue

            n = self.config.N_NOISE_SAMPLES
            noise_scale = self.config.NOISE_SCALE_FRAC * clean_norm
            noise = (
                torch.randn(n, act.shape[0], device=act.device, dtype=act.dtype)
                * noise_scale
            )
            noisy = act.unsqueeze(0) + noise
            ratios = (noisy.norm(p=2, dim=1) / clean_norm).detach().cpu().tolist()
            collapsed = [r < collapse_thresh for r in ratios]

            results.append(
                {
                    "clean_norm": clean_norm,
                    "avg_norm_ratio": float(np.mean(ratios)),
                    "min_norm_ratio": float(np.min(ratios)),
                    "norm_ratios": ratios,
                    "collapse_threshold": collapse_thresh,
                    "noise_scale": noise_scale,
                    "deception_detected": any(collapsed),
                    "n_collapsed": int(sum(collapsed)),
                }
            )

        while len(results) < len(prompts):
            results.append({"error": "Sentinel activation missing for sample"})

        return results


def _build_synthetic_vectors(model, layers: List[int]) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(42)
    vectors: Dict[str, np.ndarray] = {}
    hidden = model.config.hidden_size
    for layer_idx in layers:
        if layer_idx >= len(model.model.layers):
            continue
        vec = rng.randn(hidden).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        vectors[f"layer_{layer_idx}"] = vec
    print(f"Using synthetic steering vectors for {len(vectors)} layers (test mode)")
    return vectors


def format_mmlu_prompt(question: str, choices: List[str]) -> str:
    prompt = f"{question}\n"
    for idx, choice in enumerate(choices):
        prompt += f"{CHOICE_LABELS[idx]}. {choice}\n"
    prompt += "\nAnswer with only the letter (A, B, C, or D)."
    return prompt


def extract_answer_letter(response: str) -> Optional[str]:
    resp = response.strip().upper()

    if resp in CHOICE_LABELS:
        return resp

    patterns = [
        r"\b([A-D])\b\)?\.?\s*$",
        r"(?:ANSWER|OPTION)\s*(?:IS|:)?\s*\(?([A-D])\)?",
        r"^\(?([A-D])\)?[\.\,\:\s]",
        r"\(([A-D])\)",
    ]
    for pat in patterns:
        match = re.search(pat, resp, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    match = re.search(r"\b([A-D])\b", resp)
    if match:
        return match.group(1).upper()

    return None


def _normalize_mmlu_item(item: Dict[str, Any]) -> Dict[str, Any]:
    if "question" not in item or "choices" not in item or "answer" not in item:
        raise ValueError("Each MMLU item must contain question, choices, answer fields")

    choices = item["choices"]
    if not isinstance(choices, list) or len(choices) != 4:
        raise ValueError("Each MMLU item must contain exactly 4 choices")

    answer = item["answer"]
    if isinstance(answer, int):
        answer_idx = answer
    elif isinstance(answer, str):
        answer = answer.strip().upper()
        if answer in CHOICE_LABELS:
            answer_idx = CHOICE_LABELS.index(answer)
        elif answer.isdigit():
            answer_idx = int(answer)
        else:
            raise ValueError(f"Unrecognized answer value: {answer}")
    else:
        raise ValueError(f"Unsupported answer type: {type(answer).__name__}")

    if answer_idx < 0 or answer_idx > 3:
        raise ValueError(f"Answer index out of range: {answer_idx}")

    return {
        "question": str(item["question"]),
        "subject": str(item.get("subject", "unknown_subject")),
        "choices": [str(c) for c in choices],
        "answer": answer_idx,
    }


def load_mmlu_from_json(mmlu_path: str, max_samples: Optional[int] = None):
    if not os.path.exists(mmlu_path):
        raise FileNotFoundError(
            f"MMLU file not found: {mmlu_path}. "
            "Place balanced-mmlu-questions-across-subjects.json in phase11_data/mmlu/."
        )

    with open(mmlu_path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        if "data" in raw and isinstance(raw["data"], list):
            data = raw["data"]
        elif "questions" in raw and isinstance(raw["questions"], list):
            data = raw["questions"]
        else:
            raise ValueError("JSON dict must include list under 'data' or 'questions'")
    elif isinstance(raw, list):
        data = raw
    else:
        raise ValueError(f"Expected JSON list/dict, got {type(raw).__name__}")

    norm_data = [_normalize_mmlu_item(item) for item in data]

    if max_samples is not None and max_samples > 0:
        norm_data = norm_data[:max_samples]

    subjects = defaultdict(list)
    for item in norm_data:
        subjects[item["subject"]].append(item)

    print(f"Loaded {len(norm_data)} MMLU questions across {len(subjects)} subjects")
    return norm_data, dict(subjects)


def load_mmlu_from_hf(
    dataset_name: str,
    config_name: str,
    split_name: str,
    max_samples: Optional[int] = None,
):
    if load_dataset is None:
        raise RuntimeError(
            "datasets package is not installed. Install with: pip install datasets"
        )

    print(
        f"Loading MMLU from Hugging Face: dataset={dataset_name}, "
        f"config={config_name}, split={split_name}"
    )

    try:
        ds = load_dataset(dataset_name, config_name, split=split_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load HF MMLU dataset {dataset_name}/{config_name}:{split_name}: {exc}"
        )

    rows = []
    for item in ds:
        row = {
            "question": item.get("question", ""),
            "subject": item.get("subject", config_name),
            "choices": item.get("choices", []),
            "answer": item.get("answer", None),
        }
        rows.append(_normalize_mmlu_item(row))

    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]

    subjects = defaultdict(list)
    for item in rows:
        subjects[item["subject"]].append(item)

    print(f"Loaded {len(rows)} MMLU questions across {len(subjects)} subjects (HF)")
    return rows, dict(subjects)


def _save_json(path: str, payload: Dict[str, Any]):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def run_sentinel_mmlu_benchmark(
    model,
    pipeline: SentinelMMLUPipeline,
    tokenizer,
    config: Config,
    subjects_dict: Dict[str, List[Dict[str, Any]]],
    checkpoint_path: str,
    resume: bool,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {"baseline": {}, "steered": {}, "sentinel": {}}
    progress = {
        "subject_idx": 0,
        "question_idx": 0,
        "batch_size": config.BATCH_SIZE,
    }

    if resume and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                if "results" in loaded and isinstance(loaded["results"], dict):
                    results = loaded["results"]
                if "progress" in loaded and isinstance(loaded["progress"], dict):
                    progress.update(loaded["progress"])
                print(f"Resuming Phase 11 from checkpoint: {checkpoint_path}")
        except Exception as exc:
            print(f"WARNING: Could not load checkpoint ({exc}). Starting fresh.")

    sorted_subjects = sorted(subjects_dict.keys())
    total_questions = sum(len(subjects_dict[s]) for s in sorted_subjects)

    done_questions = 0
    for subject in sorted_subjects:
        base_len = len(results.get("baseline", {}).get(subject, {}).get("data", []))
        steer_len = len(results.get("steered", {}).get(subject, {}).get("data", []))
        sent_len = len(results.get("sentinel", {}).get(subject, {}).get("data", []))
        done_questions += min(base_len, steer_len, sent_len)

    start_done = done_questions
    run_start = time.time()

    def save_checkpoint(subject_idx: int, question_idx: int, batch_size: int):
        payload = {
            "results": results,
            "progress": {
                "subject_idx": subject_idx,
                "question_idx": question_idx,
                "batch_size": batch_size,
            },
            "meta": {
                "total_questions": total_questions,
            },
        }
        _save_json(checkpoint_path, payload)

    print(
        f"\nRunning Sentinel+MMLU benchmark: subjects={len(sorted_subjects)}, "
        f"questions={total_questions}"
    )

    for subject_idx in range(progress["subject_idx"], len(sorted_subjects)):
        subject = sorted_subjects[subject_idx]
        questions = subjects_dict[subject]
        n_questions = len(questions)

        base_existing = list(
            results.get("baseline", {}).get(subject, {}).get("data", [])
        )
        steer_existing = list(
            results.get("steered", {}).get(subject, {}).get("data", [])
        )
        sent_existing = list(
            results.get("sentinel", {}).get(subject, {}).get("data", [])
        )

        aligned_len = min(len(base_existing), len(steer_existing), len(sent_existing))
        if len(base_existing) != aligned_len:
            base_existing = base_existing[:aligned_len]
        if len(steer_existing) != aligned_len:
            steer_existing = steer_existing[:aligned_len]
        if len(sent_existing) != aligned_len:
            sent_existing = sent_existing[:aligned_len]

        start_q = (
            progress["question_idx"]
            if subject_idx == progress["subject_idx"]
            else aligned_len
        )
        start_q = max(start_q, aligned_len)

        if start_q >= n_questions:
            continue

        print("\n" + "=" * 60)
        print(
            f"[{subject_idx + 1}/{len(sorted_subjects)}] subject={subject} "
            f"questions={n_questions}"
            + (f" [resuming from {start_q}]" if start_q > 0 else "")
        )
        print("=" * 60)

        baseline_data = base_existing
        steered_data = steer_existing
        sentinel_data = sent_existing

        baseline_correct = int(sum(1 for row in baseline_data if row.get("correct")))
        steered_correct = int(sum(1 for row in steered_data if row.get("correct")))
        sentinel_detected = int(
            sum(1 for row in sentinel_data if row.get("deception_detected", False))
        )

        batch_size = max(1, int(progress.get("batch_size", config.BATCH_SIZE)))
        idx = start_q

        while idx < n_questions:
            batch = questions[idx : idx + batch_size]
            prompts = [
                format_mmlu_prompt(item["question"], item["choices"]) for item in batch
            ]

            try:
                pipeline.disable_steering()
                baseline_responses = generate_responses_batched(
                    model, tokenizer, prompts, config
                )

                gate_scores = pipeline.compute_gate_scores_batched(
                    model, tokenizer, prompts
                )
                gate_scales = pipeline.get_gated_alpha_scales(gate_scores)

                pipeline.enable_steering(gate_scales)
                steered_responses = generate_responses_batched(
                    model, tokenizer, prompts, config
                )

                sentinel_rows = pipeline.run_sentinel_prefill_batched(
                    model,
                    tokenizer,
                    prompts,
                    gate_scales,
                )

            except torch.cuda.OutOfMemoryError:
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    print(
                        f"  [OOM] Reducing batch size to {batch_size} and retrying..."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise RuntimeError("OOM at batch size 1. Cannot continue.")

            for item, b_resp, s_resp, g_cos, g_scale, sent in zip(
                batch,
                baseline_responses,
                steered_responses,
                gate_scores,
                gate_scales,
                sentinel_rows,
            ):
                gold_letter = CHOICE_LABELS[item["answer"]]

                b_pred = extract_answer_letter(b_resp)
                b_ok = b_pred == gold_letter
                if b_ok:
                    baseline_correct += 1

                s_pred = extract_answer_letter(s_resp)
                s_ok = s_pred == gold_letter
                if s_ok:
                    steered_correct += 1

                detected = bool(sent.get("deception_detected", False))
                if detected:
                    sentinel_detected += 1

                question_text = (
                    item["question"][:120] + "..."
                    if len(item["question"]) > 120
                    else item["question"]
                )

                baseline_data.append(
                    {
                        "question": question_text,
                        "gold": gold_letter,
                        "predicted": b_pred,
                        "correct": b_ok,
                        "raw_response": b_resp[:200],
                    }
                )

                steered_data.append(
                    {
                        "question": question_text,
                        "gold": gold_letter,
                        "predicted": s_pred,
                        "correct": s_ok,
                        "raw_response": s_resp[:200],
                        "gate_cos": float(g_cos),
                        "gate_scale": float(g_scale),
                        "alpha_effective": float(config.ALPHA_PEAK * g_scale),
                    }
                )

                sentinel_data.append(
                    {
                        "question": question_text,
                        "gate_cos": float(g_cos),
                        "gate_scale": float(g_scale),
                        "clean_norm": float(sent.get("clean_norm", 0.0)),
                        "avg_norm_ratio": float(sent.get("avg_norm_ratio", 1.0)),
                        "min_norm_ratio": float(sent.get("min_norm_ratio", 1.0)),
                        "n_collapsed": int(sent.get("n_collapsed", 0)),
                        "deception_detected": detected,
                        "norm_ratios": sent.get("norm_ratios", []),
                        "error": sent.get("error"),
                    }
                )

            idx += len(batch)
            done_questions += len(batch)

            b_acc = baseline_correct / len(baseline_data) if baseline_data else 0.0
            s_acc = steered_correct / len(steered_data) if steered_data else 0.0
            d_rate = sentinel_detected / len(sentinel_data) if sentinel_data else 0.0
            avg_ratio = (
                float(np.mean([r.get("avg_norm_ratio", 1.0) for r in sentinel_data]))
                if sentinel_data
                else 1.0
            )

            results["baseline"][subject] = {
                "data": baseline_data,
                "metrics": {
                    "correct": baseline_correct,
                    "total": len(baseline_data),
                    "accuracy": b_acc,
                },
            }
            results["steered"][subject] = {
                "data": steered_data,
                "metrics": {
                    "correct": steered_correct,
                    "total": len(steered_data),
                    "accuracy": s_acc,
                },
            }
            results["sentinel"][subject] = {
                "data": sentinel_data,
                "metrics": {
                    "detected": sentinel_detected,
                    "total": len(sentinel_data),
                    "detection_rate": d_rate,
                    "avg_norm_ratio": avg_ratio,
                },
            }

            save_checkpoint(subject_idx, idx, batch_size)

            elapsed = time.time() - run_start
            run_done = done_questions - start_done
            rate = run_done / elapsed if elapsed > 0 and run_done > 0 else 0.0
            rem_min = (
                ((total_questions - done_questions) / rate / 60) if rate > 0 else 0.0
            )
            done_subject = len(baseline_data)

            if done_subject <= 2 or done_subject % max(10, config.BATCH_SIZE) == 0:
                print(
                    f"  [{done_questions}/{total_questions}] {subject} "
                    f"{done_subject}/{n_questions} "
                    f"acc_b={b_acc*100:.1f}% acc_s={s_acc*100:.1f}% "
                    f"det={d_rate*100:.1f}% batch={len(batch)} rem={rem_min:.1f}m"
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        b_acc_final = baseline_correct / n_questions if n_questions else 0.0
        s_acc_final = steered_correct / n_questions if n_questions else 0.0
        d_rate_final = sentinel_detected / n_questions if n_questions else 0.0
        avg_ratio_final = (
            float(np.mean([r.get("avg_norm_ratio", 1.0) for r in sentinel_data]))
            if sentinel_data
            else 1.0
        )

        results["baseline"][subject] = {
            "data": baseline_data,
            "metrics": {
                "correct": baseline_correct,
                "total": n_questions,
                "accuracy": b_acc_final,
            },
        }
        results["steered"][subject] = {
            "data": steered_data,
            "metrics": {
                "correct": steered_correct,
                "total": n_questions,
                "accuracy": s_acc_final,
            },
        }
        results["sentinel"][subject] = {
            "data": sentinel_data,
            "metrics": {
                "detected": sentinel_detected,
                "total": n_questions,
                "detection_rate": d_rate_final,
                "avg_norm_ratio": avg_ratio_final,
            },
        }

        save_checkpoint(subject_idx + 1, 0, config.BATCH_SIZE)
        print(
            f"  -> final subject metrics: baseline={b_acc_final*100:.1f}% "
            f"steered={s_acc_final*100:.1f}% sentinel_detect={d_rate_final*100:.1f}%"
        )

    pipeline.disable_steering()
    return results


def create_plots(
    results: Dict[str, Dict[str, Any]], plots_dir: str, config: Config
) -> Dict[str, Any]:
    os.makedirs(plots_dir, exist_ok=True)

    subjects = sorted(results.get("baseline", {}).keys())
    if not subjects:
        print("No results to plot.")
        return {}

    acc_b = [results["baseline"][s]["metrics"]["accuracy"] for s in subjects]
    acc_s = [results["steered"][s]["metrics"]["accuracy"] for s in subjects]
    det_r = [
        results["sentinel"][s]["metrics"].get("detection_rate", 0.0) for s in subjects
    ]
    avg_nr = [
        results["sentinel"][s]["metrics"].get("avg_norm_ratio", 1.0) for s in subjects
    ]

    short_names = [s.replace("_", " ").title()[:24] for s in subjects]
    y = np.arange(len(subjects))

    total_b = sum(results["baseline"][s]["metrics"]["correct"] for s in subjects)
    total_s = sum(results["steered"][s]["metrics"]["correct"] for s in subjects)
    total_n = sum(results["baseline"][s]["metrics"]["total"] for s in subjects)
    total_det = sum(
        results["sentinel"][s]["metrics"].get("detected", 0) for s in subjects
    )

    overall_b = total_b / total_n if total_n else 0.0
    overall_s = total_s / total_n if total_n else 0.0
    overall_delta = overall_s - overall_b
    overall_det_rate = total_det / total_n if total_n else 0.0
    overall_avg_ratio = float(np.mean(avg_nr)) if avg_nr else 1.0

    fig, axes = plt.subplots(2, 2, figsize=(20, max(8, len(subjects) * 0.24 + 4)))

    width = 0.35
    ax = axes[0, 0]
    ax.barh(
        y - width / 2,
        acc_b,
        width,
        label="Baseline",
        color="#90CAF9",
        edgecolor="#1565C0",
    )
    ax.barh(
        y + width / 2,
        acc_s,
        width,
        label="Steered + Sentinel Gate",
        color="#42A5F5",
        edgecolor="#0D47A1",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1.05)
    ax.set_title("MMLU Per-Subject Accuracy")
    ax.axvline(x=0.25, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    ax = axes[0, 1]
    bars = ax.bar(
        ["Baseline", "Steered"],
        [overall_b, overall_s],
        color=["#90CAF9", "#42A5F5"],
        edgecolor=["#1565C0", "#0D47A1"],
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall Accuracy")
    ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, [overall_b, overall_s]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val*100:.1f}%",
            ha="center",
            fontsize=11,
        )

    ax.text(
        0.02,
        0.02,
        f"delta = {overall_delta*100:+.2f}%\n"
        f"sentinel_detect = {overall_det_rate*100:.1f}%\n"
        f"avg_norm_ratio = {overall_avg_ratio:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.8},
    )

    ax = axes[1, 0]
    deltas = [s - b for b, s in zip(acc_b, acc_s)]
    colors = [
        "#EF5350" if d < -0.05 else "#66BB6A" if d > 0.05 else "#BDBDBD" for d in deltas
    ]
    ax.barh(y, deltas, color=colors, edgecolor="gray", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel("Delta (Steered - Baseline)")
    ax.set_title("Per-Subject Accuracy Delta")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.axvline(
        x=-config.ACCURACY_COLLAPSE_THRESHOLD,
        color="red",
        linestyle="--",
        alpha=0.6,
        label=f"Collapse ({config.ACCURACY_COLLAPSE_THRESHOLD*100:.0f}%)",
    )
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.barh(y, det_r, color="#FFB74D", edgecolor="#E65100", alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Sentinel Detection Rate")
    ax.set_title("Per-Subject Sentinel Detection")

    ax2 = ax.twiny()
    ax2.plot(avg_nr, y, "o-", color="#37474F", markersize=4, linewidth=1.2)
    ax2.set_xlabel("Avg Norm Ratio")
    collapse_ratio = 1.0 / config.SENTINEL_COLLAPSE_THRESHOLD
    ax2.axvline(collapse_ratio, color="red", linestyle="--", alpha=0.5)
    ax2.axvline(1.0, color="green", linestyle="-", alpha=0.3)

    plt.tight_layout()
    out_path = f"{plots_dir}/26_sentinel_mmlu_benchmark.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "overall_baseline_accuracy": overall_b,
        "overall_steered_accuracy": overall_s,
        "overall_accuracy_delta": overall_delta,
        "overall_sentinel_detection_rate": overall_det_rate,
        "overall_avg_norm_ratio": overall_avg_ratio,
        "accuracy_collapse_detected": overall_delta
        < -config.ACCURACY_COLLAPSE_THRESHOLD,
    }

    print(f"Saved: {out_path}")
    print("\nPhase 11 summary:")
    print(f"  Baseline overall: {overall_b*100:.1f}% ({total_b}/{total_n})")
    print(f"  Steered overall:  {overall_s*100:.1f}% ({total_s}/{total_n})")
    print(f"  Delta:            {overall_delta*100:+.2f}%")
    print(f"  Sentinel detect:  {overall_det_rate*100:.1f}% ({total_det}/{total_n})")
    print(f"  Avg norm ratio:   {overall_avg_ratio:.3f}")

    if summary["accuracy_collapse_detected"]:
        print(
            "  ACCURACY COLLAPSE DETECTED "
            f"(threshold: {config.ACCURACY_COLLAPSE_THRESHOLD*100:.0f}%)"
        )
    else:
        print(
            "  No accuracy collapse "
            f"(threshold: {config.ACCURACY_COLLAPSE_THRESHOLD*100:.0f}%)"
        )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Phase 11: Sentinel-Gated MMLU Benchmark (Server)"
    )

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--data-dir", type=str, default=Config.DATA_DIR)
    parser.add_argument("--mmlu-path", type=str, default=Config.MMLU_PATH)
    parser.add_argument("--output-dir", type=str, default=Config.OUTPUT_ROOT)
    parser.add_argument("--plots-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)

    parser.add_argument(
        "--vector-source",
        type=str,
        default="disentangled",
        choices=["disentangled", "ttpd"],
    )

    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=Config.MAX_NEW_TOKENS)
    parser.add_argument(
        "--max-samples", type=int, default=0, help="0 means full dataset"
    )

    parser.add_argument("--sigma", type=float, default=Config.SIGMA)
    parser.add_argument("--alpha-peak", type=float, default=Config.ALPHA_PEAK)
    parser.add_argument("--peak-layer", type=int, default=Config.PEAK_LAYER)

    parser.add_argument("--gate-layer", type=int, default=Config.GATE_LAYER)
    parser.add_argument("--gate-threshold", type=float, default=Config.GATE_THRESHOLD)
    parser.add_argument("--gate-sharpness", type=float, default=Config.GATE_SHARPNESS)

    parser.add_argument("--sentinel-layer", type=int, default=Config.SENTINEL_LAYER)
    parser.add_argument(
        "--noise-scale-frac", type=float, default=Config.NOISE_SCALE_FRAC
    )
    parser.add_argument("--noise-samples", type=int, default=Config.N_NOISE_SAMPLES)
    parser.add_argument(
        "--sentinel-collapse-threshold",
        type=float,
        default=Config.SENTINEL_COLLAPSE_THRESHOLD,
    )

    parser.add_argument(
        "--accuracy-collapse-threshold",
        type=float,
        default=Config.ACCURACY_COLLAPSE_THRESHOLD,
    )

    parser.add_argument("--hf-dataset", type=str, default="cais/mmlu")
    parser.add_argument("--hf-config", type=str, default="all")
    parser.add_argument("--hf-split", type=str, default="test")
    parser.add_argument("--allow-hf-fallback", action="store_true")
    parser.add_argument("--force-hf-refresh", action="store_true")

    parser.add_argument("--resume", action="store_true")

    if "ipykernel" in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    config = Config()
    config.DATA_DIR = args.data_dir
    config.MMLU_PATH = args.mmlu_path

    config.BATCH_SIZE = max(1, args.batch_size)
    config.MAX_NEW_TOKENS = max(1, args.max_new_tokens)

    config.SIGMA = float(args.sigma)
    config.ALPHA_PEAK = float(args.alpha_peak)
    config.PEAK_LAYER = int(args.peak_layer)

    config.GATE_LAYER = int(args.gate_layer)
    config.GATE_THRESHOLD = float(args.gate_threshold)
    config.GATE_SHARPNESS = float(args.gate_sharpness)

    config.SENTINEL_LAYER = int(args.sentinel_layer)
    config.NOISE_SCALE_FRAC = float(args.noise_scale_frac)
    config.N_NOISE_SAMPLES = max(1, int(args.noise_samples))
    config.SENTINEL_COLLAPSE_THRESHOLD = float(args.sentinel_collapse_threshold)

    config.ACCURACY_COLLAPSE_THRESHOLD = float(args.accuracy_collapse_threshold)

    output_root = args.output_dir
    results_dir = args.results_dir or f"{output_root}/results"
    plots_dir = args.plots_dir or f"{output_root}/plots"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    max_samples = None if args.max_samples <= 0 else args.max_samples
    if args.test:
        max_samples = 80

    print("=" * 70)
    print("PHASE 11: SENTINEL-GATED MMLU BENCHMARK")
    print("=" * 70)
    print(f"Data dir: {config.DATA_DIR}")
    print(f"MMLU path: {config.MMLU_PATH}")
    print(f"Output root: {output_root}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Max new tokens: {config.MAX_NEW_TOKENS}")
    print(f"Samples: {'FULL' if max_samples is None else max_samples}")
    print(f"Vector source: {args.vector_source}")
    print(
        f"Steering: alpha={config.ALPHA_PEAK}, sigma={config.SIGMA}, peak={config.PEAK_LAYER}"
    )
    print(
        f"Gate: layer={config.GATE_LAYER}, tau={config.GATE_THRESHOLD}, "
        f"sharpness={config.GATE_SHARPNESS}"
    )
    print(
        f"Sentinel: layer={config.SENTINEL_LAYER}, noise_frac={config.NOISE_SCALE_FRAC}, "
        f"noise_samples={config.N_NOISE_SAMPLES}, collapse={config.SENTINEL_COLLAPSE_THRESHOLD}"
    )
    print(
        "HF fallback: "
        f"{args.allow_hf_fallback} | force refresh: {args.force_hf_refresh}"
    )

    print("\n[1/5] Loading model...")
    model, tokenizer, device = load_model(config, test_mode=args.test)

    print("\n[2/5] Loading MMLU dataset...")
    if args.allow_hf_fallback and args.force_hf_refresh:
        all_data, subjects_dict = load_mmlu_from_hf(
            dataset_name=args.hf_dataset,
            config_name=args.hf_config,
            split_name=args.hf_split,
            max_samples=max_samples,
        )
        try:
            os.makedirs(Path(config.MMLU_PATH).parent, exist_ok=True)
            with open(config.MMLU_PATH, "w") as f:
                json.dump(all_data, f, indent=2)
            print(f"Saved HF fallback dataset to {config.MMLU_PATH}")
        except Exception as exc:
            print(f"WARNING: Could not save fallback MMLU JSON locally: {exc}")
    elif os.path.exists(config.MMLU_PATH):
        all_data, subjects_dict = load_mmlu_from_json(
            config.MMLU_PATH, max_samples=max_samples
        )
    elif args.allow_hf_fallback:
        all_data, subjects_dict = load_mmlu_from_hf(
            dataset_name=args.hf_dataset,
            config_name=args.hf_config,
            split_name=args.hf_split,
            max_samples=max_samples,
        )
        try:
            os.makedirs(Path(config.MMLU_PATH).parent, exist_ok=True)
            with open(config.MMLU_PATH, "w") as f:
                json.dump(all_data, f, indent=2)
            print(f"Saved HF fallback dataset to {config.MMLU_PATH}")
        except Exception as exc:
            print(f"WARNING: Could not save fallback MMLU JSON locally: {exc}")
    else:
        raise FileNotFoundError(
            f"MMLU JSON missing at {config.MMLU_PATH}. "
            "Provide the file, or rerun with --allow-hf-fallback."
        )

    print("\n[3/5] Loading steering vectors...")
    if args.test and not os.path.exists(
        f"{config.DATA_DIR}/steering_vectors_{args.vector_source}.npz"
    ):
        vectors = _build_synthetic_vectors(model, Config.ALL_LAYERS)
    else:
        vectors = load_steering_vectors(
            config.DATA_DIR, args.vector_source, Config.ALL_LAYERS
        )

    print("\n[4/5] Initializing pipeline + running benchmark...")
    pipeline = SentinelMMLUPipeline(
        model=model,
        steering_vectors=vectors,
        config=config,
        device=device,
    )
    pipeline.register_hooks()

    checkpoint_path = f"{results_dir}/phase11_sentinel_mmlu_checkpoint_in_progress.json"
    results = run_sentinel_mmlu_benchmark(
        model=model,
        pipeline=pipeline,
        tokenizer=tokenizer,
        config=config,
        subjects_dict=subjects_dict,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
    )

    print("\n[5/5] Saving outputs and plots...")
    summary = create_plots(results, plots_dir, config)

    results_path = (
        f"{results_dir}/phase11_sentinel_mmlu_results_{args.vector_source}.json"
    )
    final_payload = {
        "metadata": {
            "model": config.MODEL_NAME,
            "vector_source": args.vector_source,
            "batch_size": config.BATCH_SIZE,
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "samples": "full" if max_samples is None else max_samples,
            "alpha_peak": config.ALPHA_PEAK,
            "sigma": config.SIGMA,
            "peak_layer": config.PEAK_LAYER,
            "gate_layer": config.GATE_LAYER,
            "gate_threshold": config.GATE_THRESHOLD,
            "gate_sharpness": config.GATE_SHARPNESS,
            "sentinel_layer": pipeline.sentinel_layer_actual,
            "noise_scale_frac": config.NOISE_SCALE_FRAC,
            "noise_samples": config.N_NOISE_SAMPLES,
            "sentinel_collapse_threshold": config.SENTINEL_COLLAPSE_THRESHOLD,
            "accuracy_collapse_threshold": config.ACCURACY_COLLAPSE_THRESHOLD,
        },
        "summary": summary,
        "results": results,
    }
    _save_json(results_path, final_payload)

    pipeline.remove_hooks()
    del model, tokenizer, pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("PHASE 11 COMPLETE")
    print("=" * 70)
    print(f"Results JSON: {results_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Plots dir: {plots_dir}")


if __name__ == "__main__":
    main()
