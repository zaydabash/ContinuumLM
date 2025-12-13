#!/usr/bin/env julia
"""
    standalone_train.jl

Standalone training script that doesn't rely on the NeuralODELM module loading.
Directly includes the necessary source files for training.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Required for Data.jl serialization - must come before includes
using Base.Serialization

# Directly include the source files we need
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
    config_path = length(ARGS) >= 1 ? ARGS[1] : "config/small_debug.toml"

    println("Loading config from: $config_path")
    cfg = Config.load_config(config_path)

    println("Loading corpus from: $(cfg.data.corpus_path)")
    corpus = Data.load_corpus(cfg.data.corpus_path)

    println("Building tokenizer...")
    tok = Data.build_tokenizer(corpus; vocab_size=cfg.data.vocab_size)

    # Skip saving tokenizer for now to avoid serialization issues
    println("Tokenizer built (not saved due to serialization issues)")

    println("Encoding corpus...")
    seqs = Data.encode_corpus(tok, corpus; seq_len=cfg.training.seq_len)

    # Split into train/val
    train_seqs, val_seqs = Data.split_train_val(seqs, cfg.data.train_split)
    println("Train sequences: $(length(train_seqs)), Val sequences: $(length(val_seqs))")

    train_batches = Data.make_batches(train_seqs, cfg.training.batch_size)
    val_batches = length(val_seqs) > 0 ? Data.make_batches(val_seqs, cfg.training.batch_size) : nothing

    println("Building model...")
    model = Models.build_model(cfg.model)
    println("Model type: $(typeof(model.core_block))")

    println("Starting training...")
    Training.train!(model, train_batches, val_batches, cfg)

    println("Training completed!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
