# Dynamic Activation Steering for Deceptive Alignment

> **Representation Engineering with Gaussian Depth Scheduling and Sentinel Protocols**

Deceptive alignment occurs when advanced AI systems behave perfectly safely during alignment training (e.g., RLHF or Constitutional AI) but harbor hidden misaligned objectives for deployment. Unlike traditional exploits, deception is externally invisible, concealed entirely within the model’s internal neural representations.

This repository contains the codebase, benchmarks, and experimental results for **Dynamic Activation Steering (DAS)**, an inference-time, weight-preserving framework designed to detect and mitigate deceptive alignment behaviors in Large Language Models (LLMs).

## 🚀 Key Contributions & Features

1. **Mapping the Deception (Contrastive Dataset)**
   We use a contrastive dataset (honest vs. scheming responses) and force the model to externalize its reasoning into a private `<scratchpad>`. This yields exceptional separation of honest and deceptive intent (Cohen's $d = 3.16$ at Layer 16) before output filters can mask it.

2. **Untangling Truth from Phrasing (TTPD Geometry)**
   Standard Difference-in-Means (DIM) vectors often conflate honest intent with grammatical polarity (e.g., affirmative "Yes" vs. negated "No"). Using Negation Disentanglement and Ordinary Least Squares (OLS), we isolate the pure truth direction, achieving **93.7% accuracy** at Layer 16 with near-zero entanglement ($\cos(t_G, t_P) \approx -0.07$).

3. **Defeating the Hydra Effect (Gaussian Depth Scheduling)**
   Blindly applying uniform activation steering causes the model's fluency to collapse. We implement **Gaussian Depth Scheduling**, applying the intervention as a bell-curve centered at the model’s "Honesty Peak" (Layer 16). This maintains stable steering performance (up to $\sim 0.47$) where uniform steering fails.

4. **Preserving Capability (Dynamic Gating)**
   To ensure a zero capability tax on benign prompts, we apply a sigmoid gating mechanism. The gate evaluates the cosine similarity of early-layer activations (Layer 6) against the truth vector. It achieves **zero MMLU capability degradation** (0.558 gated vs. 0.550 baseline) by leaving safe queries untouched.

5. **Real-Time Detection (Sentinel Protocol)**
   Deceptive states exhibit topological fragility. We inject 1% Gaussian noise at the "Deception Horizon" (Layer 31) to detect these unstable states. Responses exceeding an $\ell_2$-norm ratio threshold ($r > 1.0075$) are flagged and halted *before* generation reaches the user.

## 📂 Repository Structure

- `benchmarks/`: Evaluation scripts and datasets for measuring model honesty and performance, including MMLU, BeHonest, TruthfulQA, and MASK.
- `outputs/`: Generated results, plots, graphs, and metric comparisons mapping the performance of gating, Gaussian steering, and uniform steering sweeps.
- `phases/`: Code detailing the incremental experimental phases (dataset generation, vector extraction, steering setup, gating implementation, and Sentinel protocol).
- `utils/`: Core utilities for steering operations, metric computation, and model wrappers.

## 📊 Evaluation & Results

The DAS framework was evaluated on Llama-3-8B-Instruct (4-bit NF4 quantization) across multiple dimensions:

- **TruthfulQA**: Honest intervention improves factual correctness from **0.411** (baseline) to **0.476** (steered with dynamic gating).
- **MASK Benchmark**: Overall honesty scores improved from **0.416** (baseline) to **0.480** (gated steering).
- **MMLU (Capabilities)**: Pure uniform steering reduces MMLU accuracy to $0.492$. With Dynamic Gating, task performance is recovered and slightly surpasses the baseline (**0.558** vs **0.550**).
- **Sentinel Protocol**: Actively detects deception across 8 categories (Sycophancy, Instrumental Convergence, Capability Hiding, etc.), achieving up to **70-80% recall** in identifying long-horizon deception and goal preservation strategies.

## ⚙️ How It Works

1. **Activation Extraction**: Prompts are passed through the model, and activations are isolated over the `<scratchpad>` tokens.

2. **Steering (Forward Pass Modification)**: For identified deceptive queries, the truth vector $\theta_{true}$ is added to the residual stream with a depth-wise Gaussian scaling factor $\alpha_L$:

$$
x'_L = x_L + \alpha_L \cdot \theta_{true}
$$

3. **Detection**: At Layer 31, noise is injected. If the activation norm explodes:

$$
\frac{\lVert x_{31} + \epsilon \rVert_2}{\lVert x_{31} \rVert_2} > 1.0075
$$

generation is aborted.
   
## 📜 Citation & Credits

Authors: Shail Shah, Chaitanya Shah, Tathya Vaghasia, Soham Sadavarte, Kavya Bhalodi.
