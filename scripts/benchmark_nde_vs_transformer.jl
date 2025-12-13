#!/usr/bin/env julia
"""
    benchmark_nde_vs_transformer.jl

Compare Neural ODE vs Discrete Transformer performance.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

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

function benchmark_model(model_config, training_config, data_config, model_name)
    println("\n=== $model_name Benchmark ===")

    # Load data
    corpus = Data.load_corpus(data_config.corpus_path)
    tok = Data.build_tokenizer(corpus; vocab_size=data_config.vocab_size)
    seqs = Data.encode_corpus(tok, corpus; seq_len=training_config.seq_len)

    train_seqs, val_seqs = Data.split_train_val(seqs, 0.9)
    train_batches = Data.make_batches(train_seqs, training_config.batch_size)
    val_batches = Data.make_batches(val_seqs, training_config.batch_size)

    # Build model
    println("Building $model_name...")
    model = Models.build_model(model_config)

    # Quick training test
    num_steps = 20  # Short test
    test_config = Config.ConfigBundle(
        model=model_config,
        training=Config.TrainingConfig(
            batch_size=training_config.batch_size,
            seq_len=training_config.seq_len,
            num_steps=num_steps,
            log_every=10,
            eval_every=20,
            lr=training_config.lr,
            device="cpu"
        ),
        data=data_config
    )

    println("Training $model_name for $num_steps steps...")
    start_time = time()
    Training.train!(model, train_batches, val_batches, test_config)
    training_time = time() - start_time

    # Single forward pass timing
    x, y = first(train_batches)
    forward_times = Float64[]
    for _ in 1:5
        t = @elapsed model(x)
        push!(forward_times, t * 1000)  # ms
    end
    avg_forward_time = mean(forward_times)

    println("$model_name Results:")
    println("  Training time: $(round(training_time, digits=2))s")
    println("  Avg forward pass: $(round(avg_forward_time, digits=2))ms")
    println("  Model type: $(typeof(model.core_block))")

    return avg_forward_time, training_time
end

function main()
    println("=== Neural ODE vs Transformer Comparison ===\n")

    # Common config
    vocab_size = 1000
    seq_len = 32
    batch_size = 2
    d_model = 64
    n_heads = 2
    d_ff = 128

    data_config = Config.DataConfig(
        corpus_path="data/small_combined_corpus.txt",
        vocab_size=vocab_size
    )

    training_config = Config.TrainingConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        lr=1e-3,
        device="cpu"
    )

    # Test configurations
    configs = [
        ("Discrete Transformer (2 layers)", Config.ModelConfig(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            vocab_size=vocab_size, max_seq_len=seq_len,
            is_neural_ode=false, n_layers=2
        )),
        ("Discrete Transformer (4 layers)", Config.ModelConfig(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            vocab_size=vocab_size, max_seq_len=seq_len,
            is_neural_ode=false, n_layers=4
        )),
        ("Neural ODE (Euler)", Config.ModelConfig(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            vocab_size=vocab_size, max_seq_len=seq_len,
            is_neural_ode=true, ode_solver="Euler"
        )),
        ("Neural ODE (Tsit5)", Config.ModelConfig(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            vocab_size=vocab_size, max_seq_len=seq_len,
            is_neural_ode=true, ode_solver="Tsit5"
        ))
    ]

    results = []

    for (name, model_config) in configs
        try
            forward_time, train_time = benchmark_model(model_config, training_config, data_config, name)
            push!(results, (name, forward_time, train_time, true))
        catch e
            println("$name failed: $e")
            push!(results, (name, NaN, NaN, false))
        end
    end

    # Summary
    println("\n" * "="^60)
    println("SUMMARY")
    println("="^60)
    println("Model                          | Forward (ms) | Train (s) | Status")
    println("-" ^ 60)

    for (name, forward, train, success) in results
        status = success ? "✓" : "✗"
        forward_str = isnan(forward) ? "FAILED" : "$(round(forward, digits=1))"
        train_str = isnan(train) ? "FAILED" : "$(round(train, digits=1))"
        println("$name | $forward_str | $train_str | $status")
    end

    println("\n=== Analysis ===")
    println("• Discrete Transformers: More predictable performance, established architecture")
    println("• Neural ODEs: Continuous-time processing, potentially more parameter-efficient")
    println("• Trade-offs: ODEs may be slower per step but could learn more efficiently")
    println("• Current Status: Discrete transformers working well, ODEs need debugging")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
