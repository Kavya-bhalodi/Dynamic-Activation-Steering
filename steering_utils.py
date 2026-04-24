import os
import torch
import numpy as np
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import matplotlib

class BaseConfig:
    MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"
    INPUT_DIR = "./data"
    OUTPUT_DIR = "./output/results"
    PLOTS_DIR = "./output/plots"
    ALL_LAYERS = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 31]
    PEAK_LAYER = 16
    DEFAULT_SIGMA = 4.0
    DEFAULT_ALPHA_BASE = 10.0
    DEFAULT_SIGMA_SWEEP = [1, 2, 3, 4, 5, 6]
    DEFAULT_ALPHA_SWEEP = [0, 0.125, 0.25, 0.5, 0.75, 1]
    N_RUNS = 1
    BATCH_SIZE = 64
    MAX_NEW_TOKENS = 300
    TEMPERATURE = 0.7
    TOP_P = 0.9
    DO_SAMPLE = True
    USE_4BIT = False
    HF_TOKEN = "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"

def init_environment():
    if torch.cuda.is_available() and torch.cuda.device_count() != 1:
        raise RuntimeError(f"CRITICAL ERROR: Expected 1 GPU, but saw {torch.cuda.device_count()}. Isolation failed!")
    transformers.logging.set_verbosity_error()
    matplotlib.use('Agg')

def load_model(config, test_mode=False):
    if test_mode:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"\nLoading TEST model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        model.eval()
        return model, tokenizer, "cpu"

    print(f"\nLoading model: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if getattr(config, "USE_4BIT", False):
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME, quantization_config=quant_config, device_map="auto", token=config.HF_TOKEN
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", token=config.HF_TOKEN
        )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return model, tokenizer, device

def load_steering_vectors(data_dir: str, source: str = "disentangled", layers: List[int] = None) -> Dict[str, np.ndarray]:
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
        layers = BaseConfig.ALL_LAYERS

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

def compute_gaussian_weights(layers: List[int], peak_layer: int = 16, sigma: float = 4.0) -> Dict[int, float]:
    weights = {}
    for L in layers:
        w = np.exp(-((L - peak_layer) ** 2) / (2 * sigma ** 2))
        weights[L] = float(w)
    return weights

def compute_per_layer_alphas(layers: List[int], alpha_base: float, peak_layer: int = 16, sigma: float = 4.0) -> Dict[int, float]:
    weights = compute_gaussian_weights(layers, peak_layer, sigma)
    return {L: alpha_base * w for L, w in weights.items()}

def generate_responses_batched(model, tokenizer, prompt_list, max_new_tokens=300, temperature=0.7, top_p=0.9, do_sample=True, device="cuda"):
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompt_list, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    input_device = next(model.parameters()).device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=None,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[:, input_len:]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    del inputs, outputs
    if torch.cuda.is_available():
        import gc
        torch.cuda.empty_cache()
        gc.collect()
    return [r.strip() for r in responses]
