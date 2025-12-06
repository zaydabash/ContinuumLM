#!/usr/bin/env julia
# Smoke test: verify the module loads and can run a basic forward pass.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

println("Loading NeuralODELM...")
using NeuralODELM
using NeuralODELM.Config
using NeuralODELM.Models
using NeuralODELM.Training: lm_loss

println("Building model...")
# Tiny model config
mc = NeuralODELM.Config.ModelConfig(
    d_model=32,
    n_heads=2,
    d_ff=64,
    vocab_size=100,
    max_seq_len=16,
    is_neural_ode=true,
    ode_t0=0.0,
    ode_t1=1.0,
    ode_solver="Tsit5"
)

model = Models.build_model(mc)
println("Model built successfully: $(typeof(model))")

println("Creating dummy data...")
# Create dummy token data: (seq_len, batch)
x = rand(1:mc.vocab_size, 15, 2)  # (seq_len=15, batch=2)
y = rand(1:mc.vocab_size, 15, 2)

println("Running forward pass...")
loss = lm_loss(model, x, y)
println("✓ Forward pass successful!")
println("Loss: $loss")
@assert isfinite(loss) "Loss must be finite"
@assert loss > 0 "Loss should be positive"

println("\n✓ Smoke test passed! Module loads and forward pass works.")

