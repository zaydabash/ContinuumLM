#!/usr/bin/env julia
"""
    benchmark.jl

Performance benchmarking for Neural ODE Language Model.
Tests different ODE solvers, adjoint methods, and model configurations.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using NeuralODELM
using NeuralODELM.Config
using NeuralODELM.Models
using NeuralODELM.Training
using BenchmarkTools
using Statistics

"""
    benchmark_forward_pass(model, x; n_runs=10)

Benchmark forward pass performance.
"""
function benchmark_forward_pass(model, x; n_runs=10)
    println("Benchmarking forward pass...")

    # Warmup
    for _ in 1:3
        model(x)
    end

    # Benchmark
    times = Float64[]
    for _ in 1:n_runs
        t = @elapsed model(x)
        push!(times, t * 1000)  # Convert to milliseconds
    end

    mean_time = mean(times)
    std_time = std(times)
    println(".2f")
    return mean_time, std_time
end

"""
    benchmark_training_step(model, x, y; n_runs=10)

Benchmark a single training step (forward + backward).
"""
function benchmark_training_step(model, x, y; n_runs=10)
    println("Benchmarking training step...")

    # Warmup
    for _ in 1:3
        loss, grads = Flux.withgradient(model) do m
            lm_loss(m, x, y)
        end
    end

    # Benchmark
    times = Float64[]
    for _ in 1:n_runs
        t = @elapsed begin
            loss, grads = Flux.withgradient(model) do m
                lm_loss(m, x, y)
            end
        end
        push!(times, t * 1000)  # Convert to milliseconds
    end

    mean_time = mean(times)
    std_time = std(times)
    println(".2f")
    return mean_time, std_time
end

"""
    benchmark_memory_usage(model, x, y)

Measure memory usage for forward and backward passes.
"""
function benchmark_memory_usage(model, x, y)
    println("Benchmarking memory usage...")

    # Forward pass memory
    GC.gc()
    initial_mem = Sys.free_memory()
    logits = model(x)
    forward_mem = initial_mem - Sys.free_memory()
    println(".2f")

    # Backward pass memory
    GC.gc()
    initial_mem = Sys.free_memory()
    loss, grads = Flux.withgradient(model) do m
        lm_loss(m, x, y)
    end
    backward_mem = initial_mem - Sys.free_memory()
    println(".2f")

    return forward_mem, backward_mem
end

"""
    benchmark_different_configs()

Benchmark different model configurations.
"""
function benchmark_different_configs()
    println("\n=== Configuration Benchmarks ===\n")

    # Create test data
    seq_len = 32
    batch_size = 4
    vocab_size = 1000
    x = rand(1:vocab_size, seq_len, batch_size)
    y = rand(1:vocab_size, seq_len, batch_size)

    configs = [
        ("Small Neural ODE", ModelConfig(d_model=64, n_heads=2, d_ff=128, vocab_size=vocab_size, max_seq_len=seq_len, is_neural_ode=true, ode_solver="Euler")),
        ("Medium Neural ODE", ModelConfig(d_model=128, n_heads=4, d_ff=256, vocab_size=vocab_size, max_seq_len=seq_len, is_neural_ode=true, ode_solver="Tsit5")),
        ("Large Neural ODE", ModelConfig(d_model=256, n_heads=8, d_ff=512, vocab_size=vocab_size, max_seq_len=seq_len, is_neural_ode=true, ode_solver="Vern7")),
        ("Small Transformer", ModelConfig(d_model=64, n_heads=2, d_ff=128, vocab_size=vocab_size, max_seq_len=seq_len, is_neural_ode=false, n_layers=2)),
        ("Medium Transformer", ModelConfig(d_model=128, n_heads=4, d_ff=256, vocab_size=vocab_size, max_seq_len=seq_len, is_neural_ode=false, n_layers=4)),
        ("Large Transformer", ModelConfig(d_model=256, n_heads=8, d_ff=512, vocab_size=vocab_size, max_seq_len=seq_len, is_neural_ode=false, n_layers=6))
    ]

    results = []

    for (name, config) in configs
        println("Testing $name...")
        try
            model = Models.build_model(config)
            forward_time, _ = benchmark_forward_pass(model, x, n_runs=5)
            train_time, _ = benchmark_training_step(model, x, y, n_runs=5)

            push!(results, (name, forward_time, train_time))
            println("✓ $name completed")
        catch e
            println("✗ $name failed: $e")
            push!(results, (name, NaN, NaN))
        end
        println()
    end

    # Summary table
    println("=== Summary ===")
    println("Configuration                  | Forward (ms) | Train Step (ms)")
    println("-" ^ 60)
    for (name, forward, train) in results
        if isnan(forward)
            println("$name | FAILED      | FAILED")
        else
            println(".2f")
        end
    end
end

"""
    benchmark_ode_solvers()

Benchmark different ODE solvers.
"""
function benchmark_ode_solvers()
    println("\n=== ODE Solver Benchmarks ===\n")

    # Create test model
    config = ModelConfig(d_model=64, n_heads=2, d_ff=128, vocab_size=500, max_seq_len=16, is_neural_ode=true)
    model = Models.build_model(config)

    # Test data
    x = rand(1:500, 16, 2)
    y = rand(1:500, 16, 2)

    solvers = [
        "Euler", "RK4", "Tsit5", "Vern7", "BS3", "DP5", "KenCarp4"
    ]

    println("ODE Solver     | Forward (ms) | Train Step (ms)")
    println("-" ^ 50)

    for solver_name in solvers
        try
            # Create model with specific solver
            test_config = ModelConfig(d_model=64, n_heads=2, d_ff=128, vocab_size=500, max_seq_len=16,
                                    is_neural_ode=true, ode_solver=solver_name)
            test_model = Models.build_model(test_config)

            forward_time, _ = benchmark_forward_pass(test_model, x, n_runs=3)
            train_time, _ = benchmark_training_step(test_model, x, y, n_runs=3)

            println(".2f")
        catch e
            println("$solver_name | FAILED       | FAILED")
        end
    end
end

"""
    benchmark_generation(model, tokenizer, prompt; max_tokens=50)

Benchmark text generation performance.
"""
function benchmark_generation(model, tokenizer, prompt="The"; max_tokens=50)
    println("\n=== Generation Benchmark ===\n")

    device_fn = NeuralODELM.Utils.select_device("auto")
    model = device_fn(model)

    println("Benchmarking text generation...")
    println("Prompt: \"$prompt\"")
    println("Max tokens: $max_tokens")

    # Warmup
    for _ in 1:2
        NeuralODELM.Generation.generate_text(model, tokenizer, prompt, max_new_tokens=10)
    end

    # Benchmark
    times = Float64[]
    for _ in 1:5
        t = @elapsed NeuralODELM.Generation.generate_text(model, tokenizer, prompt, max_new_tokens=max_tokens)
        push!(times, t * 1000)  # Convert to milliseconds
    end

    mean_time = mean(times)
    tokens_per_sec = max_tokens / (mean_time / 1000)

    println(".2f")
    println(".1f")

    return mean_time, tokens_per_sec
end

function main()
    println("=== Neural ODE Language Model Benchmarks ===\n")

    if length(ARGS) >= 1 && ARGS[1] == "--configs"
        benchmark_different_configs()
    elseif length(ARGS) >= 1 && ARGS[1] == "--solvers"
        benchmark_ode_solvers()
    elseif length(ARGS) >= 1 && ARGS[1] == "--generation"
        # Load a trained model for generation benchmark
        config_path = length(ARGS) >= 2 ? ARGS[2] : "config/small_debug.toml"
        checkpoint_path = length(ARGS) >= 3 ? ARGS[3] : "checkpoints/best_model.bson"

        println("Loading model from $checkpoint_path...")
        try
            cfg = NeuralODELM.Config.load_config(config_path)
            model, step, loss = NeuralODELM.Training.load_checkpoint(checkpoint_path)
            tokenizer = NeuralODELM.Data.load_tokenizer(cfg.data.tokenizer_path)

            benchmark_generation(model, tokenizer)
        catch e
            println("Failed to load model: $e")
        end
    else
        println("Usage:")
        println("  julia scripts/benchmark.jl --configs      # Benchmark different model configs")
        println("  julia scripts/benchmark.jl --solvers      # Benchmark different ODE solvers")
        println("  julia scripts/benchmark.jl --generation [config] [checkpoint]  # Benchmark text generation")
        println()
        benchmark_different_configs()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
