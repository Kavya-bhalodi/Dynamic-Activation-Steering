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
        raise RuntimeError(
            f"CRITICAL ERROR: Expected 1 GPU, but saw {torch.cuda.device_count()}. Isolation failed!"
        )
    transformers.logging.set_verbosity_error()
    matplotlib.use("Agg")


def load_model(config, test_mode=False):
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
        return (model, tokenizer, "cpu")
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
        print(f"  GPU memory: {torch.cuda.memory_allocated() / 1000000000.0:.1f} GB")
    return (model, tokenizer, device)


def load_steering_vectors(
    data_dir: str, source: str = "disentangled", layers: List[int] = None
) -> Dict[str, np.ndarray]:
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
            alt_key = (
                f"{layer_name}_theta_true_unit"
                if source == "disentangled"
                else f"{layer_name}_t_G_unit"
            )
            if alt_key in sv:
                vectors[layer_name] = sv[alt_key]
            else:
                print(f"  WARNING: No vector for {layer_name}")
    print(f"  Loaded vectors for {len(vectors)} layers")
    return vectors


def compute_gaussian_weights(
    layers: List[int], peak_layer: int = 16, sigma: float = 4.0
) -> Dict[int, float]:
    weights = {}
    for L in layers:
        w = np.exp(-((L - peak_layer) ** 2) / (2 * sigma**2))
        weights[L] = float(w)
    return weights


def compute_per_layer_alphas(
    layers: List[int], alpha_base: float, peak_layer: int = 16, sigma: float = 4.0
) -> Dict[int, float]:
    weights = compute_gaussian_weights(layers, peak_layer, sigma)
    return {L: alpha_base * w for L, w in weights.items()}


def generate_responses_batched(
    model,
    tokenizer,
    prompt_list,
    max_new_tokens=300,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    device="cuda",
):
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompt_list, return_tensors="pt", max_length=1024, truncation=True, padding=True
    )
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


class SentinelPipeline:

    def __init__(self, model, steering_vectors, config, device="cuda"):
        self.model = model
        self.config = config
        self.device = device
        self.hooks = []
        self.steering_tensors = {}
        for layer_idx in config.ALL_LAYERS:
            key = f"layer_{layer_idx}"
            if key in steering_vectors:
                vec = steering_vectors[key]
                if np.linalg.norm(vec) > 1e-08:
                    self.steering_tensors[layer_idx] = torch.tensor(
                        vec, dtype=torch.float32, device=device
                    )
        gate_key = f"layer_{config.GATE_LAYER}"
        if gate_key in steering_vectors:
            gv = steering_vectors[gate_key]
            norm = np.linalg.norm(gv)
            self.gate_truth_dir = torch.tensor(
                gv / norm if norm > 1e-08 else gv, dtype=torch.float32, device=device
            )
        else:
            self.gate_truth_dir = None
        self.layer_alphas = compute_per_layer_alphas(
            config.ALL_LAYERS, config.ALPHA_PEAK, config.PEAK_LAYER, config.SIGMA
        )
        self.steering_active = False
        self.current_alpha_scale = 1.0
        self.sentinel_activation = None

    def _create_steering_hook(self, layer_idx):
        steering_vec = self.steering_tensors[layer_idx]

        def hook_fn(module, input, output):
            if not self.steering_active:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            alpha_l = self.layer_alphas[layer_idx] * self.current_alpha_scale
            perturbation = (alpha_l * steering_vec).to(
                device=hidden.device, dtype=hidden.dtype
            )
            if hidden.dim() == 3:
                modified = hidden + perturbation.unsqueeze(0).unsqueeze(0)
            else:
                modified = hidden + perturbation.unsqueeze(0)
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return hook_fn

    def _create_sentinel_hook(self):

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() == 3:
                self.sentinel_activation = hidden[0, -1, :].detach().clone()
            else:
                self.sentinel_activation = hidden[-1, :].detach().clone()
            return output

        return hook_fn

    def register_hooks(self):
        self.remove_hooks()
        for layer_idx in self.config.ALL_LAYERS:
            if layer_idx not in self.steering_tensors:
                continue
            layer = self.model.model.layers[layer_idx]
            self.hooks.append(
                layer.register_forward_hook(self._create_steering_hook(layer_idx))
            )
        sentinel_idx = self.config.SENTINEL_LAYER
        n_layers = len(self.model.model.layers)
        if sentinel_idx >= n_layers:
            sentinel_idx = n_layers - 1
            print(
                f"  [Sentinel] Adjusted to layer {sentinel_idx} (model has {n_layers} layers)"
            )
        self.sentinel_layer_actual = sentinel_idx
        sentinel_layer = self.model.model.layers[sentinel_idx]
        self.hooks.append(
            sentinel_layer.register_forward_hook(self._create_sentinel_hook())
        )

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def enable_steering(self):
        self.steering_active = True

    def disable_steering(self):
        self.steering_active = False

    def compute_gate_score(self, model, tokenizer, prompt):
        if self.gate_truth_dir is None:
            return 0.0
        was_active = self.steering_active
        self.steering_active = False
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=1024, truncation=True
        )
        input_device = next(model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        captured = {}

        def hook_fn(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            if hidden.dim() == 3:
                captured["act"] = hidden[0, -1, :].detach().clone()
            else:
                captured["act"] = hidden[-1, :].detach().clone()

        hk = model.model.layers[self.config.GATE_LAYER].register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**inputs)
        hk.remove()
        self.steering_active = was_active
        del inputs
        torch.cuda.empty_cache()
        act = captured["act"].float()
        truth = self.gate_truth_dir.float().to(act.device)
        return F.cosine_similarity(act.unsqueeze(0), truth.unsqueeze(0)).item()

    def get_gated_alpha_scale(self, cos_sim):
        tau = self.config.GATE_THRESHOLD
        k = self.config.GATE_SHARPNESS
        x = -k * (cos_sim - tau)
        return 1.0 / (1.0 + np.exp(-x))

    def run_sentinel_test(self, model, tokenizer, prompt):
        self.sentinel_activation = None
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=1024, truncation=True
        )
        input_device = next(model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
        del inputs
        torch.cuda.empty_cache()
        if self.sentinel_activation is None:
            return {
                "error": "No sentinel activation captured",
                "deception_detected": False,
                "clean_norm": 0.0,
                "avg_norm_ratio": 1.0,
            }
        act = self.sentinel_activation.float()
        clean_norm = act.norm(p=2).item()
        if clean_norm < 1e-10:
            return {
                "clean_norm": 0.0,
                "deception_detected": False,
                "avg_norm_ratio": 1.0,
                "norm_ratios": [],
            }
        N = self.config.N_NOISE_SAMPLES
        noise_scale = self.config.NOISE_SCALE_FRAC * clean_norm
        noise = (
            torch.randn(N, act.shape[0], device=act.device, dtype=act.dtype)
            * noise_scale
        )
        noisy_acts = act.unsqueeze(0) + noise
        noisy_norms = noisy_acts.norm(p=2, dim=1)
        norm_ratios = (noisy_norms / clean_norm).cpu().tolist()
        collapse_thresh = 1.0 / self.config.COLLAPSE_THRESHOLD_SENTINEL
        collapsed = [r < collapse_thresh for r in norm_ratios]
        deception_detected = any(collapsed)
        return {
            "clean_norm": clean_norm,
            "avg_norm_ratio": float(np.mean(norm_ratios)),
            "min_norm_ratio": float(min(norm_ratios)),
            "norm_ratios": norm_ratios,
            "collapse_threshold": collapse_thresh,
            "noise_scale": noise_scale,
            "deception_detected": deception_detected,
            "n_collapsed": sum(collapsed),
        }


class GaussianDepthSteerer:

    def __init__(
        self,
        model,
        steering_vectors: Dict[str, np.ndarray],
        alpha_base: float = 0.0,
        peak_layer: int = 16,
        sigma: float = 3.0,
        layers: Optional[List[int]] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.alpha_base = alpha_base
        self.peak_layer = peak_layer
        self.sigma = sigma
        self.layers = layers or Config.ALL_LAYERS
        self.device = device
        self.hooks: List[Any] = []
        self.active = False
        self.hidden_size = model.config.hidden_size
        self.layer_alphas = compute_per_layer_alphas(
            self.layers, alpha_base, peak_layer, sigma
        )
        self.steering_tensors: Dict[int, torch.Tensor] = {}
        for layer_idx in self.layers:
            key = f"layer_{layer_idx}"
            if key not in steering_vectors:
                continue
            vec = np.asarray(steering_vectors[key]).flatten()
            norm = np.linalg.norm(vec)
            if norm <= 1e-08:
                continue
            if vec.shape[0] != self.hidden_size:
                if vec.shape[0] > self.hidden_size:
                    print(
                        f"  WARNING: {key} has {vec.shape[0]} dims; truncating to {self.hidden_size}"
                    )
                    vec = vec[: self.hidden_size]
                else:
                    print(
                        f"  WARNING: {key} has {vec.shape[0]} dims; expected {self.hidden_size}. Skipping layer."
                    )
                    continue
            self.steering_tensors[layer_idx] = torch.tensor(
                vec, dtype=torch.float32, device=device
            )

    def _create_hook(self, layer_idx: int):
        steering_vec = self.steering_tensors[layer_idx]
        layer_alpha = self.layer_alphas[layer_idx]
        expected_hidden_size = self.hidden_size

        def hook_fn(module, hook_input, output):
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
            if hidden_states.shape[-1] != expected_hidden_size:
                return output
            vec = steering_vec.to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
            if hidden_states.ndim == 3:
                vec = vec.view(1, 1, expected_hidden_size)
            elif hidden_states.ndim == 2:
                vec = vec.view(1, expected_hidden_size)
            else:
                return output
            modified = hidden_states + layer_alpha * vec
            if modified.shape != hidden_states.shape:
                return output
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return hook_fn

    def register_hooks(self):
        self.remove_hooks()
        for layer_idx in self.layers:
            if layer_idx not in self.steering_tensors:
                continue
            layer = self.model.model.layers[layer_idx]
            self.hooks.append(layer.register_forward_hook(self._create_hook(layer_idx)))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def update_schedule(
        self, alpha_base: Optional[float] = None, sigma: Optional[float] = None
    ):
        if alpha_base is not None:
            self.alpha_base = alpha_base
        if sigma is not None:
            self.sigma = sigma
        self.layer_alphas = compute_per_layer_alphas(
            self.layers, self.alpha_base, self.peak_layer, self.sigma
        )
        self.register_hooks()

    def enable(self):
        self.active = True

    def disable(self):
        self.active = False


class DynamicGate:

    def __init__(
        self,
        model,
        truth_vector_early: np.ndarray,
        gate_layer: int = 14,
        threshold: Optional[float] = None,
        sharpness: float = 10.0,
        device: str = "cuda",
    ):
        self.model = model
        self.gate_layer = gate_layer
        self.threshold = threshold
        self.sharpness = sharpness
        vec = np.asarray(truth_vector_early).flatten()
        norm = np.linalg.norm(vec)
        if norm > 1e-08:
            vec = vec / norm
        self.truth_dir = torch.tensor(vec, dtype=torch.float32, device=device)

    def extract_gate_activation(
        self,
        model,
        tokenizer,
        prompt: str,
        steerer: Optional[GaussianDepthSteerer] = None,
    ):
        was_active = steerer.active if steerer else False
        if steerer:
            steerer.disable()
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=1024, truncation=True
        )
        input_device = next(model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        captured: Dict[str, torch.Tensor] = {}

        def hook_fn(module, hook_input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden):
                return
            if hidden.dim() == 3:
                captured["act"] = hidden[0, -1, :].detach().clone()
            else:
                captured["act"] = hidden[-1, :].detach().clone()

        hook = model.model.layers[self.gate_layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            model(**inputs)
        hook.remove()
        if steerer and was_active:
            steerer.enable()
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if "act" not in captured:
            raise RuntimeError("Failed to capture gate activation")
        return captured["act"]

    def compute_gate_score(self, activation: torch.Tensor) -> float:
        act = activation.float().unsqueeze(0)
        truth = self.truth_dir.float().to(activation.device).unsqueeze(0)
        return float(torch.nn.functional.cosine_similarity(act, truth).item())

    def get_effective_alpha(self, cos_sim: float, alpha_peak: float) -> float:
        if self.threshold is None:
            return float(alpha_peak)
        x = -self.sharpness * (cos_sim - self.threshold)
        gate_value = 1.0 / (1.0 + np.exp(-x))
        return float(alpha_peak * gate_value)

    def calibrate_threshold(
        self,
        model,
        tokenizer,
        prompts: List[Dict[str, Any]],
        steerer: Optional[GaussianDepthSteerer] = None,
        percentile: int = 50,
    ) -> List[float]:
        print(f"\nCalibrating gate threshold on layer {self.gate_layer}...")
        cos_sims = []
        for prompt_info in prompts:
            act = self.extract_gate_activation(
                model, tokenizer, prompt_info["prompt"], steerer
            )
            score = self.compute_gate_score(act)
            cos_sims.append(score)
            print(f"  {prompt_info['id']:<20} cos_sim={score:+.4f}")
        self.threshold = float(np.percentile(cos_sims, percentile))
        print(
            f"Gate threshold set to {self.threshold:.4f} (percentile={percentile}, range=[{min(cos_sims):.4f}, {max(cos_sims):.4f}])"
        )
        return cos_sims


def compute_honesty_score(
    response: str, honesty_keywords: List[str], deception_keywords: List[str]
) -> Dict[str, float]:
    response_lower = response.lower()
    honest_hits = sum((1 for kw in honesty_keywords if kw.lower() in response_lower))
    deceptive_hits = sum(
        (1 for kw in deception_keywords if kw.lower() in response_lower)
    )
    total = honest_hits + deceptive_hits
    score = (honest_hits - deceptive_hits) / total if total > 0 else 0.0
    return {
        "honesty_hits": honest_hits,
        "deception_hits": deceptive_hits,
        "total_keywords": total,
        "honesty_score": score,
    }


def compute_quality_score(
    honesty_score: float, avg_length: float, min_length: int = 150
) -> float:
    length_factor = min(1.0, avg_length / min_length)
    return float(honesty_score * length_factor)


class SentinelBeHonestPipeline:

    def __init__(
        self,
        model,
        steering_vectors: Dict[str, np.ndarray],
        config: Config,
        device: str,
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
            config.ALL_LAYERS, config.ALPHA_PEAK, config.PEAK_LAYER, config.SIGMA
        )
        self.steering_tensors: Dict[int, torch.Tensor] = {}
        for layer_idx in config.ALL_LAYERS:
            key = f"layer_{layer_idx}"
            if key not in steering_vectors:
                continue
            vec = np.asarray(steering_vectors[key]).flatten()
            norm = np.linalg.norm(vec)
            if norm <= 1e-08:
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
            if gate_vec.shape[0] > self.hidden_size:
                gate_vec = gate_vec[: self.hidden_size]
            gate_norm = np.linalg.norm(gate_vec)
            if gate_norm > 1e-08 and gate_vec.shape[0] == self.hidden_size:
                self.gate_truth_dir = torch.tensor(
                    gate_vec / gate_norm, dtype=torch.float32, device=device
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
                        dtype=hidden_states.dtype, device=hidden_states.device
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
                        dtype=hidden_states.dtype, device=hidden_states.device
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
        sentinel_idx = min(self.config.SENTINEL_LAYER, n_layers - 1)
        self.sentinel_layer_actual = sentinel_idx
        sentinel_layer = self.model.model.layers[sentinel_idx]
        self.hooks.append(
            sentinel_layer.register_forward_hook(self._create_sentinel_hook())
        )

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def set_alpha_schedule(self, alpha_peak: float, sigma: Optional[float] = None):
        self.config.ALPHA_PEAK = alpha_peak
        if sigma is not None:
            self.config.SIGMA = sigma
        self.layer_alphas = compute_per_layer_alphas(
            self.config.ALL_LAYERS,
            self.config.ALPHA_PEAK,
            self.config.PEAK_LAYER,
            self.config.SIGMA,
        )

    def compute_batch_gate_scales(
        self, model, tokenizer, prompts: List[str]
    ) -> torch.Tensor:
        if not prompts:
            return torch.ones(0, dtype=torch.float32)
        if self.gate_truth_dir is None:
            return torch.ones(len(prompts), dtype=torch.float32)
        was_active = self.steering_active
        self.steering_active = False
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            [_format_prompt(p) for p in prompts],
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True,
        )
        input_device = next(model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        captured: Dict[str, torch.Tensor] = {}

        def hook_fn(module, hook_input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden):
                return
            if hidden.ndim == 3:
                captured["act"] = hidden[:, -1, :].detach().clone()
            elif hidden.ndim == 2:
                captured["act"] = hidden.detach().clone()

        gate_hook = model.model.layers[self.config.GATE_LAYER].register_forward_hook(
            hook_fn
        )
        with torch.no_grad():
            model(**inputs)
        gate_hook.remove()
        self.steering_active = was_active
        if "act" not in captured:
            return torch.ones(len(prompts), dtype=torch.float32)
        act = captured["act"].float()
        truth = self.gate_truth_dir.float().to(act.device).view(1, -1)
        cos_sim = F.cosine_similarity(act, truth.expand_as(act), dim=1)
        x = -self.config.GATE_SHARPNESS * (cos_sim - self.config.GATE_THRESHOLD)
        scales = 1.0 / (1.0 + torch.exp(-x))
        return scales.detach().cpu()

    def run_batch_sentinel_test(self) -> List[Dict[str, Any]]:
        if self.sentinel_activations is None:
            return [{"error": "No sentinel activation captured"}]
        acts = self.sentinel_activations.float()
        batch_results: List[Dict[str, Any]] = []
        collapse_thresh = 1.0 / self.config.SENTINEL_COLLAPSE_THRESHOLD
        for i in range(acts.shape[0]):
            act = acts[i]
            clean_norm = act.norm(p=2).item()
            if clean_norm < 1e-10:
                batch_results.append(
                    {
                        "clean_norm": 0.0,
                        "avg_norm_ratio": 1.0,
                        "min_norm_ratio": 1.0,
                        "norm_ratios": [1.0],
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
            noisy_norms = noisy.norm(p=2, dim=1)
            ratios = (noisy_norms / clean_norm).detach().cpu().tolist()
            collapsed = [r < collapse_thresh for r in ratios]
            batch_results.append(
                {
                    "clean_norm": clean_norm,
                    "avg_norm_ratio": float(np.mean(ratios)),
                    "min_norm_ratio": float(min(ratios)),
                    "norm_ratios": ratios,
                    "collapse_threshold": collapse_thresh,
                    "noise_scale": noise_scale,
                    "deception_detected": any(collapsed),
                    "n_collapsed": int(sum(collapsed)),
                }
            )
        return batch_results


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
            config.ALL_LAYERS, config.ALPHA_PEAK, config.PEAK_LAYER, config.SIGMA
        )
        self.steering_tensors: Dict[int, torch.Tensor] = {}
        for layer_idx in config.ALL_LAYERS:
            key = f"layer_{layer_idx}"
            if key not in steering_vectors:
                continue
            vec = np.asarray(steering_vectors[key]).flatten()
            norm = np.linalg.norm(vec)
            if norm <= 1e-08:
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
            if gate_norm > 1e-08 and gate_vec.shape[0] >= self.hidden_size:
                gate_vec = gate_vec[: self.hidden_size]
                self.gate_truth_dir = torch.tensor(
                    gate_vec / gate_norm, dtype=torch.float32, device=device
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
                        dtype=hidden_states.dtype, device=hidden_states.device
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
                        dtype=hidden_states.dtype, device=hidden_states.device
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
        self, model, tokenizer, prompts: List[str]
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
        self, model, tokenizer, prompts: List[str], gate_scales: List[float]
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
