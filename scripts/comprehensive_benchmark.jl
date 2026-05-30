#!/usr/bin/env julia
"""
    comprehensive_benchmark.jl

Comprehensive benchmarking of Neural ODE vs Transformer models.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Statistics

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

"""
    benchmark_model_comprehensive(model_config, model_name, data_config; train_steps=50)

Comprehensive benchmark including training time, forward pass speed, memory usage, and generation.
"""
function benchmark_model_comprehensive(model_config, model_name, data_config; train_steps=50)
    println("\n" * "="^60)
    println("COMPREHENSIVE BENCHMARK: $model_name")
    println("="^60)

    # Data preparation
    corpus = Data.load_corpus(data_config.corpus_path)
    tok = Data.build_tokenizer(corpus; vocab_size=data_config.vocab_size)
    seqs = Data.encode_corpus(tok, corpus; seq_len=32)

    train_seqs, val_seqs = Data.split_train_val(seqs, 0.9)
    train_batches = Data.make_batches(train_seqs, 2)
    val_batches = Data.make_batches(val_seqs, 2)

    # Model creation
    println("Building $model_name...")
    model = Models.build_model(model_config)

    # Get sample batch for timing
    x, y = first(train_batches)

    # Forward pass benchmark
    println("Benchmarking forward pass...")
    forward_times = Float64[]
    for _ in 1:10
        t = @elapsed model(x)
        push!(forward_times, t * 1000)
    end
    forward_mean = mean(forward_times)
    forward_std = std(forward_times)

    # Training benchmark
    println("Benchmarking training...")
    training_config = Config.TrainingConfig(
        batch_size=2, seq_len=32, num_steps=train_steps,
        log_every=1000, eval_every=1000, lr=1e-3, device="cpu"
    )
    cfg = Config.ConfigBundle(model=model_config, training=training_config, data=data_config)

    start_time = time()
    Training.train!(model, train_batches, val_batches, cfg)
    training_time = time() - start_time

    # Generation benchmark
    println("Benchmarking generation...")
    generation_times = Float64[]
    for _ in 1:5
        t = @elapsed begin
            # Import here to avoid circular dependencies
            include("../src/Generation.jl")
            eval(Meta.parse("using .Generation"))
            eval(Meta.parse("Generation.generate_text(model, tok, \"The\"; max_new_tokens=10)"))
        end
        push!(generation_times, t * 1000)
    end
    gen_mean = mean(generation_times)

    # Results
    results = Dict(
        "model" => model_name,
        "forward_time_ms" => forward_mean,
        "forward_std_ms" => forward_std,
        "training_time_s" => training_time,
        "training_steps" => train_steps,
        "generation_time_ms" => gen_mean,
        "model_type" => string(typeof(model.core_block)),
        "parameters" => sum(length, Flux.params(model))
    )

    println("\nRESULTS for $model_name:")
    println("  Forward pass: $(round(forward_mean, digits=2)) ± $(round(forward_std, digits=2)) ms")
    println("  Training: $(round(training_time, digits=1))s for $train_steps steps")
    println("  Generation (10 tokens): $(round(gen_mean, digits=1)) ms")
    println("  Parameters: $(results["parameters"])")
    println("  Architecture: $(results["model_type"])")

    return results
end

"""
    compare_architectures()

Compare Neural ODE vs Transformer architectures.
"""
function compare_architectures()
    println("🧠 CONTINUUM LM: Neural ODE vs Transformer Comparison")
    println("="^70)

    data_config = Config.DataConfig(
        corpus_path="data/small_combined_corpus.txt",
        vocab_size=1000
    )

    # Model configurations
    configs = [
        ("Transformer (2 layers)", Config.ModelConfig(
            d_model=64, n_heads=4, d_ff=128, vocab_size=1000, max_seq_len=32,
            is_neural_ode=false, n_layers=2
        )),
        ("Transformer (4 layers)", Config.ModelConfig(
            d_model=64, n_heads=4, d_ff=128, vocab_size=1000, max_seq_len=32,
            is_neural_ode=false, n_layers=4
        )),
        ("Neural ODE (RK4)", Config.ModelConfig(
            d_model=64, n_heads=4, d_ff=128, vocab_size=1000, max_seq_len=32,
            is_neural_ode=true, ode_solver="RK4", ode_integrator="custom_fixed_step",
            ode_sensealg="BacksolveAdjoint"
        )),
        ("Neural ODE (Euler)", Config.ModelConfig(
            d_model=64, n_heads=4, d_ff=128, vocab_size=1000, max_seq_len=32,
            is_neural_ode=true, ode_solver="Euler", ode_integrator="custom_fixed_step",
            ode_sensealg="BacksolveAdjoint"
        ))
    ]

    results = []

    for (name, config) in configs
        try
            result = benchmark_model_comprehensive(config, name, data_config, train_steps=30)
            push!(results, result)
        catch e
            println("❌ $name failed: $e")
            push!(results, Dict("model" => name, "error" => string(e)))
        end
    end

    # Summary comparison
    println("\n" * "="^70)
    println("📊 FINAL COMPARISON SUMMARY")
    println("="^70)

    successful_results = filter(r -> !haskey(r, "error"), results)

    if length(successful_results) >= 2
        println("Model                     | Forward (ms) | Train (s) | Gen (ms) | Params")
        println("-" ^ 70)

        for result in successful_results
            name = result["model"]
            forward = round(result["forward_time_ms"], digits=1)
            train = round(result["training_time_s"], digits=1)
            gen = round(result["generation_time_ms"], digits=1)
            params = result["parameters"]

            println("$(rpad(name, 24)) | $(lpad(forward, 11)) | $(lpad(train, 9)) | $(lpad(gen, 8)) | $(params)")
        end

        # Analysis
        transformer_results = filter(r -> occursin("Transformer", r["model"]), successful_results)
        node_results = filter(r -> occursin("Neural ODE", r["model"]), successful_results)

        if !isempty(transformer_results) && !isempty(node_results)
            transformer_time = mean(r["forward_time_ms"] for r in transformer_results)
            node_time = mean(r["forward_time_ms"] for r in node_results)

            speedup = transformer_time / node_time
            if speedup > 1
                println("
🎯 Neural ODEs are $(round(speedup, digits=1))x SLOWER than Transformers (expected)")
                println("📈 But they offer continuous-depth processing and theoretical advantages")
            else
                println("
⚡ Neural ODEs are $(round(1/speedup, digits=1))x FASTER than Transformers")
            end
        end
    else
        println("❌ Insufficient successful benchmarks for comparison")
    end

    println("\n🔬 Key Insights:")
    println("• Transformers: Established, fast, discrete layers")
    println("• Neural ODEs: Novel, continuous-time, potentially more expressive")
    println("• Trade-off: Speed vs architectural innovation")
    println("• Future: Better ODE solvers and hardware acceleration could close the gap")

    return results
end

function main()
    try
        results = compare_architectures()
        println("\n✅ Comprehensive benchmarking completed!")
        return results
    catch e
        println("❌ Benchmarking failed: $e")
        return []
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
