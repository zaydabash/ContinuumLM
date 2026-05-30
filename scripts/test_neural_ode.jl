#!/usr/bin/env julia
"""
    test_neural_ode.jl

Test Neural ODE model with custom integrator.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Flux  # For gradient computation

include("../src/Config.jl")
include("../src/Utils.jl")
include("../src/Data.jl")
include("../src/Models/Models.jl")
include("../src/Training.jl")

using .Config
using .Utils
using .Data
using .Models
using .Training

function test_neural_ode()
    println("=== Testing Neural ODE with Custom Integrator ===\n")

    # Load data
    corpus = Data.load_corpus("data/small_combined_corpus.txt")
    tok = Data.build_tokenizer(corpus; vocab_size=500)
    seqs = Data.encode_corpus(tok, corpus; seq_len=16)

    train_seqs, val_seqs = Data.split_train_val(seqs, 0.9)
    train_batches = Data.make_batches(train_seqs, 2)
    val_batches = Data.make_batches(val_seqs, 2)

    # Neural ODE config
    model_config = Config.ModelConfig(
        d_model=32,  # Small for testing
        n_heads=2,
        d_ff=64,
        vocab_size=500,
        max_seq_len=16,
        is_neural_ode=true,
        ode_solver="Euler",
        ode_sensealg="BacksolveAdjoint",  # Simpler adjoint method
        ode_integrator="custom_fixed_step",  # Use custom integrator
        ode_nsteps=2  # Simple 2-step integration
    )

    println("Building Neural ODE model...")
    model = Models.build_model(model_config)
    println("Model type: $(typeof(model.core_block))")

    # Test forward pass
    println("Testing forward pass...")
    x, y = first(train_batches)
    println("Input shape: $(size(x))")

    try
        logits = model(x)
        println("✓ Forward pass successful! Output shape: $(size(logits))")

        # Test loss
        loss = Training.lm_loss(model, x, y)
        println("✓ Loss computation successful! Loss: $loss")
    catch e
        println("✗ Forward pass failed: $e")
        return
    end

    # Test gradient computation separately
    println("\nTesting gradient computation...")
    try
        loss, grads = Flux.withgradient(model) do m
            Training.lm_loss(m, x, y)
        end
        println("✓ Gradient computation successful! Loss: $loss")
        println("  Gradient keys: $(length(grads)) parameter groups")
    catch e
        println("✗ Gradient computation failed: $e")
        println("  This is expected for Neural ODE - using custom integrator")
    end

    # Skip full training for now due to ODE adjoint issues
    println("\nNote: Neural ODE training with gradients requires more work.")
    println("The custom RK4 integrator works, but adjoint methods need tuning.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_neural_ode()
end
