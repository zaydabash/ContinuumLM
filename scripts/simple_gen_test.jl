#!/usr/bin/env julia
"""
    simple_gen_test.jl

Simple test for text generation.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include("../src/Config.jl")
include("../src/Utils.jl")
include("../src/Data.jl")
include("../src/Models/Models.jl")
include("../src/Training.jl")
include("../src/Generation.jl")

using .Config
using .Utils
using .Data
using .Models
using .Training
using .Generation

function simple_test()
    # Load trained model
    checkpoint_path = "checkpoints/best_model.bson"
    if !isfile(checkpoint_path)
        println("No checkpoint found")
        return
    end

    println("Loading model...")
    model, step, loss = Training.load_checkpoint(checkpoint_path)
    println("Model loaded from step $step")

    # Load tokenizer
    corpus = Data.load_corpus("data/small_combined_corpus.txt")
    tok = Data.build_tokenizer(corpus; vocab_size=2000)

    # Test simple generation
    prompt = "The"
    println("\nPrompt: \"$prompt\"")

    try
        result = Generation.generate_text(model, tok, prompt; max_new_tokens=5)
        println("Generated: \"$result\"")
        println("✓ Generation successful!")
    catch e
        println("✗ Generation failed: $e")
    end
end

simple_test()
