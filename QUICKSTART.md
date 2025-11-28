# Quick Start Guide

## Installation

1. **Install Julia 1.10+** from [julialang.org](https://julialang.org/downloads/)

2. **Clone and setup**:
```bash
cd ContinuumLM
julia --project=.
```

3. **Install dependencies**:
```julia
using Pkg
Pkg.instantiate()
```

This will install all required packages:
- Flux.jl (deep learning)
- DifferentialEquations.jl (ODE solving)
- DiffEqFlux.jl (Neural ODE integration)
- Tokenizers.jl (text tokenization)
- And others...

## First Training Run

1. **Use the provided sample corpus** (or replace `data/corpus.txt` with your own text)

2. **Train a small debug model** (fast, CPU-friendly):
```bash
julia scripts/train_neural_ode_lm.jl config/small_debug.toml
```

This will:
- Build a tokenizer from the corpus
- Train a tiny Neural ODE model (~1000 steps)
- Save checkpoints to `checkpoints/`

3. **Generate text**:
```bash
julia scripts/generate.jl config/small_debug.toml checkpoints/best_model.bson "The future"
```

## Running Tests

```bash
julia --project=. test/runtests.jl
```

## Next Steps

- **Larger model**: Use `config/neural_ode_transformer.toml`
- **Baseline comparison**: Use `config/base_transformer.toml` (discrete layers)
- **Custom data**: Replace `data/corpus.txt` with your corpus
- **GPU training**: Set `device = "gpu"` in config (requires CUDA)

## Troubleshooting

- **Package errors**: Run `Pkg.resolve()` and `Pkg.instantiate()`
- **CUDA errors**: Set `device = "cpu"` in config if GPU unavailable
- **Out of memory**: Reduce `batch_size` or `d_model` in config
- **Slow training**: Neural ODEs are slower than discrete models; use shorter `ode_t1` for faster training

## Architecture Notes

- **Neural ODE**: Continuous-depth via ODE integration (`ode_t0` to `ode_t1`)
- **Discrete**: Standard stacked Transformer layers (`n_layers`)
- Both use the same attention/feedforward blocks internally

