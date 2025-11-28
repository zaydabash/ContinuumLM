#!/usr/bin/env julia
"""
    generate.jl

Text generation script for Neural ODE Language Model.

Usage:
    julia scripts/generate.jl [config_path] [checkpoint_path] [prompt] [--max_tokens N] [--temperature T] [--top_k K] [--top_p P]

Example:
    julia scripts/generate.jl config/neural_ode_transformer.toml checkpoints/best_model.bson "Once upon a time"
"""

using Pkg
Pkg.activate(@__DIR__ * "/..")

using NeuralODELM
using NeuralODELM.Config
using NeuralODELM.Data
using NeuralODELM.Models
using NeuralODELM.Generation
using NeuralODELM.Training
using NeuralODELM.Utils

function parse_args()
    config_path = "config/neural_ode_transformer.toml"
    checkpoint_path = "checkpoints/best_model.bson"
    prompt = "Once upon a time"
    max_tokens = 100
    temperature = 1.0
    top_k = 0
    top_p = 0.0
    
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--max_tokens" && i < length(ARGS)
            max_tokens = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--temperature" && i < length(ARGS)
            temperature = parse(Float64, ARGS[i+1])
            i += 2
        elseif arg == "--top_k" && i < length(ARGS)
            top_k = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--top_p" && i < length(ARGS)
            top_p = parse(Float64, ARGS[i+1])
            i += 2
        elseif i == 1
            config_path = arg
            i += 1
        elseif i == 2
            checkpoint_path = arg
            i += 1
        elseif i == 3
            prompt = arg
            i += 1
        else
            i += 1
        end
    end
    
    return config_path, checkpoint_path, prompt, max_tokens, temperature, top_k, top_p
end

function main()
    config_path, checkpoint_path, prompt, max_tokens, temperature, top_k, top_p = parse_args()
    
    println("Loading config from: $config_path")
    cfg = load_config(config_path)
    
    println("Loading checkpoint from: $checkpoint_path")
    model, step, loss = load_checkpoint(checkpoint_path)
    
    println("Loading tokenizer...")
    tok = if isfile(cfg.data.tokenizer_path)
        load_tokenizer(cfg.data.tokenizer_path)
    else
        println("Tokenizer not found, building new one...")
        corpus = load_corpus(cfg.data.corpus_path)
        build_tokenizer(corpus; vocab_size=cfg.data.vocab_size)
    end
    
    device_fn = select_device(cfg.training.device)
    model = device_fn(model)
    
    println("\n" * "="^50)
    println("Generating text...")
    println("Prompt: $prompt")
    println("Settings: max_tokens=$max_tokens, temperature=$temperature, top_k=$top_k, top_p=$top_p")
    println("="^50 * "\n")
    
    generated = generate_text(model, tok, prompt;
                              max_new_tokens=max_tokens,
                              temperature=temperature,
                              top_k=top_k,
                              top_p=top_p)
    
    println(generated)
    println("\n" * "="^50)
end

main()

