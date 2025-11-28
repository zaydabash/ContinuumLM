#!/usr/bin/env julia
"""
    evaluate.jl

Evaluation script for Neural ODE Language Model.

Usage:
    julia scripts/evaluate.jl [config_path] [checkpoint_path]

Example:
    julia scripts/evaluate.jl config/neural_ode_transformer.toml checkpoints/best_model.bson
"""

using Pkg
Pkg.activate(@__DIR__ * "/..")

using NeuralODELM
using NeuralODELM.Config
using NeuralODELM.Data
using NeuralODELM.Models
using NeuralODELM.Evaluation
using NeuralODELM.Training
using NeuralODELM.Utils

function main()
    config_path = length(ARGS) > 0 ? ARGS[1] : "config/neural_ode_transformer.toml"
    checkpoint_path = length(ARGS) > 1 ? ARGS[2] : "checkpoints/best_model.bson"
    
    println("Loading config from: $config_path")
    cfg = load_config(config_path)
    
    println("Loading checkpoint from: $checkpoint_path")
    model, step, loss = load_checkpoint(checkpoint_path)
    
    println("Loading corpus...")
    corpus = load_corpus(cfg.data.corpus_path)
    
    println("Loading tokenizer...")
    tok = if isfile(cfg.data.tokenizer_path)
        load_tokenizer(cfg.data.tokenizer_path)
    else
        println("Tokenizer not found, building new one...")
        build_tokenizer(corpus; vocab_size=cfg.data.vocab_size)
    end
    
    println("Encoding corpus...")
    seqs = encode_corpus(tok, corpus; seq_len=cfg.training.seq_len)
    
    # Use validation split
    _, val_seqs = split_train_val(seqs, cfg.data.train_split)
    val_batches = make_batches(val_seqs, cfg.training.batch_size)
    
    device_fn = select_device(cfg.training.device)
    model = device_fn(model)
    
    println("Evaluating...")
    ppl, mean_loss = evaluate_perplexity(model, val_batches, device_fn)
    
    println("\n" * "="^50)
    println("Evaluation Results:")
    println("  Checkpoint step: $step")
    println("  Checkpoint loss: $loss")
    println("  Validation loss: $mean_loss")
    println("  Perplexity: $ppl")
    println("="^50)
end

main()

