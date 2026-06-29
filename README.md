# Neural ODE-Powered Language Model (ContinuumLM)

An implementation of a **continuous-depth Transformer language model** using Neural ODEs in Julia. This project implements a novel architecture where hidden states evolve continuously through time/depth, parameterized by Transformer-style dynamics.

## Overview

Traditional Transformers process sequences through discrete layers. This project explores an alternative: **continuous-time evolution** of hidden states via Neural ODEs. The model integrates an ODE `dh/dt = f(h, t, θ)` where `f` is parameterized by self-attention and feedforward blocks.

### Architecture

```
tokens → embeddings → Neural ODE Transformer → LM head → logits
                          ↓
                    dh/dt = TransformerBlock(h, t)
```

### Key Features

- **Continuous-depth processing** via ODE integration (DifferentialEquations.jl)
- **Proper adjoint sensitivity methods** for efficient backpropagation (InterpolatingAdjoint, BacksolveAdjoint)
- **Custom continuous-attention kernel integrator** (RK4-style fixed-step integration)
- **Reversible ODE design** for memory-efficient training
- **TensorBoard logging** for experiment tracking
- **Discrete Transformer baseline** for comparison
- **Full training pipeline** with checkpointing and validation
- **Text generation** with multiple sampling strategies (greedy, top-k, top-p)
- **GPU support** via CUDA.jl
- **Type-stable, idiomatic Julia** code
- **Comprehensive tests** and documentation

## Requirements

- Julia 1.10+
- CUDA-capable GPU (optional, but recommended for larger models)

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd ContinuumLM

# Activate Julia environment
julia --project=.

# Install dependencies
julia -e 'using Pkg; Pkg.instantiate()'
```

### Prepare Data

Create a text corpus file:

```bash
mkdir -p data
# Add your text corpus to data/corpus.txt
```

### Train a Model

**Small debug model (fast, CPU-friendly):**
```bash
julia scripts/train_neural_ode_lm.jl config/small_debug.toml
```

**Neural ODE Transformer:**
```bash
julia scripts/train_neural_ode_lm.jl config/neural_ode_transformer.toml
```

**Discrete Transformer baseline:**
```bash
julia scripts/train_neural_ode_lm.jl config/base_transformer.toml
```

### Evaluate

```bash
julia scripts/evaluate.jl config/neural_ode_transformer.toml checkpoints/best_model.bson
```

### Generate Text

```bash
julia scripts/generate.jl config/neural_ode_transformer.toml checkpoints/best_model.bson "Once upon a time"
```

With custom sampling:
```bash
julia scripts/generate.jl config/neural_ode_transformer.toml checkpoints/best_model.bson "The future of AI" --max_tokens 200 --temperature 0.8 --top_k 50 --top_p 0.9
```

### View Training Logs

Start TensorBoard to visualize training metrics:
```bash
tensorboard --logdir logs
```

Then open `http://localhost:` in your browser to view:
- Training/validation loss curves
- Perplexity metrics
- Learning rate schedule
- Gradient norms

### Run Tests

```bash
julia --project=. test/runtests.jl
```

## Project Structure

```
ContinuumLM/
├── src/
│   ├── NeuralODELM.jl          # Main module
│   ├── Config.jl                # Configuration management
│   ├── Utils.jl                 # Device selection, seeding
│   ├── Data.jl                  # Tokenization and batching
│   ├── Training.jl              # Training loop, checkpointing
│   ├── Evaluation.jl            # Perplexity, validation metrics
│   ├── Generation.jl            # Text generation utilities
│   └── Models/
│       ├── Models.jl            # Model exports
│       ├── Embeddings.jl        # Token + positional embeddings
│       ├── Attention.jl         # Multi-head self-attention
│       ├── ContinuousTransformer.jl  # Discrete stack baseline
│       ├── NeuralODEBlock.jl    # Continuous-time ODE block
│       └── LanguageModel.jl    # End-to-end LM composition
├── scripts/
│   ├── train_neural_ode_lm.jl  # Training entrypoint
│   ├── evaluate.jl              # Evaluation script
│   └── generate.jl              # Generation script
├── config/
│   ├── small_debug.toml         # Tiny model for debugging
│   ├── neural_ode_transformer.toml  # Neural ODE config
│   └── base_transformer.toml    # Discrete baseline config
├── test/
│   ├── runtests.jl              # Test suite
│   ├── test_data.jl             # Data pipeline tests
│   ├── test_models.jl           # Model component tests
│   └── test_training.jl         # Training loop tests
└── README.md                    # This file
```

## Configuration

Configuration files use TOML format. Key settings:

### Model Configuration

- `d_model`: Hidden dimension
- `n_heads`: Number of attention heads
- `d_ff`: Feedforward dimension
- `vocab_size`: Vocabulary size
- `is_neural_ode`: Use Neural ODE (true) or discrete stack (false)
- `ode_t0`, `ode_t1`: ODE integration time interval
- `ode_solver`: ODE solver ("Tsit5", "RK4", "Euler")
- `ode_sensealg`: Adjoint sensitivity method ("InterpolatingAdjoint", "BacksolveAdjoint", "QuadratureAdjoint")
- `ode_integrator`: Integration mode ("generic" or "custom_fixed_step")
- `ode_nsteps`: Number of steps for custom integrator (default: 4)
- `reversible`: Use reversible ODE for memory efficiency (default: false)
- `ode_atol`, `ode_rtol`: ODE solver tolerances

### Training Configuration

- `batch_size`: Batch size
- `seq_len`: Sequence length
- `num_steps`: Total training steps
- `lr`: Learning rate
- `weight_decay`: Weight decay for AdamW
- `grad_clip`: Gradient clipping threshold
- `warmup_steps`: Learning rate warmup steps
- `device`: "cpu", "gpu", or "auto"
- `log_dir`: Directory for TensorBoard logs (default: "logs")
- `run_name`: Name for this training run (default: "default_run")

## How It Works

### Neural ODE Block

The core innovation is the `NeuralODEBlock`, which:

1. Takes hidden state `h(t)` at depth `t`
2. Computes derivative `dh/dt = TransformerBlock(h, t)`
3. Integrates from `t=0` to `t=T` using an ODE solver
4. Returns the transformed state `h(T)`

This replaces discrete layer stacking with continuous evolution, allowing the model to learn adaptive depth.

### Advanced Features

**Adjoint Sensitivity Methods:**
- Uses `InterpolatingAdjoint` or `BacksolveAdjoint` for efficient gradient computation
- Avoids storing full forward trajectory during backpropagation
- Configurable via `ode_sensealg` in config

**Custom Continuous-Attention Kernel:**
- Optional RK4-style fixed-step integrator (`ode_integrator = "custom_fixed_step"`)
- Tailored specifically for Transformer dynamics
- Configurable number of steps via `ode_nsteps`

**Reversible ODE:**
- Memory-efficient training with `reversible = true`
- Automatically uses `BacksolveAdjoint` for optimal memory usage
- Reconstructs intermediate states on-the-fly during backprop

**Generation:**
- Autoregressive sampling via `generate_text()` with temperature, top-k, and top-p
- Full-sequence recompute at each step (KV caching is not applicable to the
  continuous-depth ODE core, which has no discrete per-layer key/value state)

### Comparison: Discrete vs Continuous

- **Discrete Transformer**: `h_{i+1} = TransformerBlock(h_i)` for `i=1..N`
- **Neural ODE**: `h(T) = h(0) + ∫₀ᵀ TransformerBlock(h(t), t) dt`

The continuous formulation can be more parameter-efficient and theoretically allows for adaptive depth.

## Results

To check whether this actually trains and whether continuous depth helps,
the Neural ODE model and the discrete Transformer baseline were trained
with **matched architecture and hyperparameters** on the real Penn
Treebank (PTB) word-level corpus (`data/corpus.txt`, ~887K words):

- `d_model=128, n_heads=4, d_ff=512, n_layers=3, seq_len=64, batch_size=8`
- `vocab_size=4000`, `lr=1e-3` (AdamW, 300-step warmup), `seed=42`
- 3000 training steps, evaluated every 500 steps on a held-out validation split
- Configs: [`config/comparison_ode.toml`](config/comparison_ode.toml), [`config/comparison_baseline.toml`](config/comparison_baseline.toml)

### Validation perplexity over training

| Step | Discrete Transformer | Neural ODE |
|------|----------------------|------------|
| 500  | 229.0                | 219.8      |
| 1000 | 166.9                | 163.2      |
| 1500 | 139.7                | 142.1      |
| 2000 | 127.2                | 132.2      |
| 2500 | 118.0                | 124.7      |
| 3000 | **113.7**            | **119.9**  |

### Non-neural reference points (same corpus, same train/val split)

| Model                          | Val. Perplexity |
|---------------------------------|-----------------|
| Unigram (frequency-only)         | 426.8           |
| Bigram (Laplace-smoothed)        | 321.5           |
| Discrete Transformer (3000 steps)| **113.7**       |
| Neural ODE (3000 steps)          | **119.9**       |

### Wall-clock cost

| Model                 | Time / training step | Time for 3000 steps |
|------------------------|----------------------|----------------------|
| Discrete Transformer    | ~0.09s               | ~4.7 min             |
| Neural ODE (RK4, 4 steps)| ~0.29s              | ~14.6 min            |

### Takeaways

- Both models clearly learn: they beat the bigram baseline by **2.8-3.2x**
  in perplexity within 3000 steps, on real text, not a toy corpus.
- At this scale and step count, the **discrete Transformer baseline edges
  out the Neural ODE variant** (113.7 vs 119.9 final val perplexity). The
  Neural ODE path is *not* free — it costs roughly **3x the wall-clock
  time per step** (RK4 with 4 substeps means 4 forward passes through the
  block per training step) without a perplexity improvement here.
- This is a single seed, single short run at modest model scale — not a
  sweep and not a claim that continuous depth never helps. It does
  establish a real, reproducible baseline number instead of an
  architecture description with no evidence behind it.
- Reproduce with: `julia --project=. scripts/train_neural_ode_lm.jl config/comparison_ode.toml`
  (swap in `comparison_baseline.toml` for the discrete baseline).

## Limitations & Future Work

This is a **research scaffold**, not a production LLM. Current limitations:

- Small model sizes (for research/education)
- Basic tokenization (word-level)
- Limited dataset support

### No incremental/cached inference for the ODE path

Generation here recomputes the full sequence at every new token (see
`generate_text` in `Generation.jl`). Standard KV caching does not transfer
to this architecture: a discrete Transformer caches per-layer keys/values
because each layer is a distinct, fixed function, but the ODE core has no
discrete layers, only a continuous trajectory `h(t)` integrated from `h(0)`
to `h(T)` — there is no fixed per-layer state to cache. This is a known
open problem for continuous-depth and attention-based ODE language models,
not something specific to this implementation. Two directions in the
literature address it:

- **ODE-RNN / Latent ODEs** (Rubanova, Chen & Duvenaud, 2019) interleave
  discrete recurrent updates at observed tokens with continuous evolution
  between them, which would let a new token update a finite recurrent state
  rather than re-running the ODE over the whole prefix.
- **Structured state-space models** (Gu et al., "S4", 2022; Gu & Dao,
  "Mamba", 2023) are also derived from continuous-time dynamical systems but
  use a linear time-invariant formulation, which has an exact equivalent
  recurrent form and supports genuine O(1)-per-token incremental inference.
  They are not attention-based and are architecturally distinct from this
  repo's Transformer-dynamics ODE, but they're the closest thing the field
  has to "Neural ODEs with caching" and the most realistic path if this
  limitation needs to be lifted rather than just documented.

**Potential extensions:**

- Larger model scales
- Advanced ODE solvers and adjoint methods
- Additional regularization techniques
- Multi-GPU training
- Integration with HuggingFace tokenizers (subword/BPE instead of word-level)

## References

- **Neural ODEs**: Chen, Rubanova, Bettencourt & Duvenaud, "Neural Ordinary Differential Equations" (NeurIPS 2018)
- **ODE-Transformer connection**: Lu et al., "Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View" (2019) — the multi-particle ODE view of Transformer depth that this repo's `dh/dt = TransformerBlock(h, t)` formulation builds on
- **Latent ODEs / ODE-RNN**: Rubanova, Chen & Duvenaud, "Latent ODEs for Irregularly-Sampled Time Series" (NeurIPS 2019) — the established approach for combining discrete recurrent state updates with continuous-time evolution, relevant to the caching limitation above
- **Structured State Space Models**: Gu, Goel & Ré, "Efficiently Modeling Long Sequences with Structured State Spaces" (S4, ICLR 2022); Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- **Continuous Normalizing Flows**: Grathwohl et al., "FFJORD" (ICLR 2019)
- **Transformers**: Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)

## License

MIT License (or as specified in your project)

## Contributing

This is a research codebase. Contributions welcome! Areas for improvement:

- Performance optimizations
- Additional ODE solvers
- Better tokenization support
- More comprehensive tests
- Documentation improvements

## Acknowledgments

Built with:
- [Flux.jl](https://fluxml.ai/) - Deep learning framework
- [DifferentialEquations.jl](https://diffeq.sciml.ai/) - ODE solving
- [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl) - Neural ODE integration

---
