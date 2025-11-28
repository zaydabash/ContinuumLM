# Neural ODE-Powered Language Model (ContinuumLM)

A research-grade implementation of a **continuous-depth Transformer language model** using Neural ODEs in Julia. This project implements a novel architecture where hidden states evolve continuously through time/depth, parameterized by Transformer-style dynamics.

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
- **KV caching** for fast autoregressive generation
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

Then open `http://localhost:6006` in your browser to view:
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

**KV Caching:**
- Efficient autoregressive generation with cached keys/values
- Avoids recomputing attention for previous tokens
- Use `generate_text_with_cache()` for faster inference

### Comparison: Discrete vs Continuous

- **Discrete Transformer**: `h_{i+1} = TransformerBlock(h_i)` for `i=1..N`
- **Neural ODE**: `h(T) = h(0) + ∫₀ᵀ TransformerBlock(h(t), t) dt`

The continuous formulation can be more parameter-efficient and theoretically allows for adaptive depth.

## Limitations & Future Work

This is a **research scaffold**, not a production LLM. Current limitations:

- Small model sizes (for research/education)
- Basic tokenization (word-level)
- Limited dataset support

**Potential extensions:**

- KV caching for Neural ODE path
- Larger model scales
- Advanced ODE solvers and adjoint methods
- Additional regularization techniques
- Multi-GPU training
- Integration with HuggingFace tokenizers

## References

- **Neural ODEs**: Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018)
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

**Note**: This project is for research and educational purposes. For production language models, consider established frameworks like Transformers.jl or PyTorch implementations.
