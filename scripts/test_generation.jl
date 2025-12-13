#!/usr/bin/env julia
"""
    test_generation.jl

Test text generation with KV caching.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include("../src/Config.jl")
include("../src/Utils.jl")
include("../src/Data.jl")
include("../src/Models/Models.jl")
include("../src/Generation.jl")
include("../src/Training.jl")

using .Config
using .Utils
using .Data
using .Models
using .Generation
using .Training

function test_generation()
    # Load the trained model
    checkpoint_path = "checkpoints/best_model.bson"
    if !isfile(checkpoint_path)
        println("No checkpoint found at $checkpoint_path")
        return
    end

    println("Loading model from $checkpoint_path...")
    model, step, loss = Training.load_checkpoint(checkpoint_path)
    println("Loaded model from step $step with loss $loss")

    # Load tokenizer
    corpus = Data.load_corpus("data/small_combined_corpus.txt")
    tok = Data.build_tokenizer(corpus; vocab_size=2000)

    # Test prompts
    prompts = [
        "The",
        "Machine learning",
        "Neural",
        "In the beginning"
    ]

    println("\n=== Testing Text Generation ===\n")

    device_fn = Utils.select_device("cpu")
    model = device_fn(model)

    for prompt in prompts
        println("="^50)
        println("Prompt: \"$prompt\"")
        println("-"^50)

        # Simple single-step generation test
        println("Testing single forward pass:")
        try
            ids = Data.encode(tok, prompt)
            x = reshape(ids, :, 1)
            logits = model(x)
            next_token_logits = logits[:, end, 1]
            next_token = argmax(next_token_logits)
            next_word = Data.decode(tok, [next_token])
            println("  Next token prediction: \"$next_word\"")
        catch e
            println("  Error: $e")
        end

        println()
    end
end

function main()
    test_generation()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
