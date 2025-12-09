#!/usr/bin/env julia
"""
    train_neural_ode_lm.jl

Training script for Neural ODE Language Model.

Usage:
    julia scripts/train_neural_ode_lm.jl [config_path]

Example:
    julia scripts/train_neural_ode_lm.jl config/small_debug.toml
"""

using Pkg
Pkg.activate(@__DIR__ * "/..")

using NeuralODELM
using NeuralODELM.Config
using NeuralODELM.Data
using NeuralODELM.Models
using NeuralODELM.Training
using NeuralODELM.Utils

function main()
    config_path = length(ARGS) > 0 ? ARGS[1] : "config/neural_ode_transformer.toml"
    
    if !isfile(config_path)
        error("Config file not found: $config_path")
    end
    
    println("Loading config from: $config_path")
    cfg = load_config(config_path)
    
    println("Loading corpus from: $(cfg.data.corpus_path)")
    corpus = load_corpus(cfg.data.corpus_path)
    
    println("Building tokenizer...")
    tok = build_tokenizer(corpus; vocab_size=cfg.data.vocab_size)
    
    # Save tokenizer for later use
    Utils.ensure_dir(dirname(cfg.data.tokenizer_path))
    save_tokenizer(tok, cfg.data.tokenizer_path)
    println("Tokenizer saved to: $(cfg.data.tokenizer_path)")
    
    println("Encoding corpus...")
    seqs = encode_corpus(tok, corpus; seq_len=cfg.training.seq_len)
    
    # Split into train/val
    train_seqs, val_seqs = split_train_val(seqs, cfg.data.train_split)
    println("Train sequences: $(length(train_seqs)), Val sequences: $(length(val_seqs))")
    
    train_batches = make_batches(train_seqs, cfg.training.batch_size)
    val_batches = length(val_seqs) > 0 ? make_batches(val_seqs, cfg.training.batch_size) : nothing
    
    println("Building model...")
    model = Models.build_model(cfg.model)
    println("Model type: $(typeof(model.core_block))")
    
    println("Starting training...")
    train!(model, train_batches, val_batches, cfg)
    
    println("Training completed!")
end

main()

