#!/usr/bin/env julia
"""
    quick_train.jl

Quick training test with the larger dataset, bypassing tokenizer save/load issues.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Manually include just what we need for training
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

function main()
    # Use the larger dataset directly
    corpus_path = "data/small_combined_corpus.txt"
    println("Loading corpus from: $corpus_path")
    corpus = Data.load_corpus(corpus_path)

    # Build tokenizer
    vocab_size = 2000  # Larger vocab for better modeling
    println("Building tokenizer with vocab_size=$vocab_size...")
    tok = Data.build_tokenizer(corpus; vocab_size=vocab_size)

    # Training config
    seq_len = 64
    batch_size = 4
    num_steps = 50  # Quick test

    println("Encoding corpus...")
    seqs = Data.encode_corpus(tok, corpus; seq_len=seq_len)

    # Split into train/val
    train_seqs, val_seqs = Data.split_train_val(seqs, 0.9)
    println("Train sequences: $(length(train_seqs)), Val sequences: $(length(val_seqs))")

    train_batches = Data.make_batches(train_seqs, batch_size)
    val_batches = length(val_seqs) > 0 ? Data.make_batches(val_seqs, batch_size) : nothing

    # Model config - test discrete Transformer first
    model_config = Config.ModelConfig(
        d_model=128,
        n_heads=4,
        d_ff=256,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        is_neural_ode=false,  # Use discrete transformer
        n_layers=4
    )

    println("Building Neural ODE model...")
    model = Models.build_model(model_config)
    println("Model type: $(typeof(model.core_block))")

    # Training config
    training_config = Config.TrainingConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        num_steps=num_steps,
        log_every=10,
        eval_every=25,
        lr=1e-3,
        device="cpu"  # Use CPU for now
    )

    cfg = Config.ConfigBundle(model=model_config, training=training_config, data=Config.DataConfig())

    println("Starting training with $num_steps steps...")
    Training.train!(model, train_batches, val_batches, cfg)

    println("Training completed!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
