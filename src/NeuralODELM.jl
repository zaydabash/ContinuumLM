"""
    NeuralODELM.jl

Neural ODE-Powered Language Model (Continuous-Time Transformer)

A research-grade implementation of a continuous-depth Transformer language model
using Neural ODEs in Julia. The model processes sequences through continuous-time
evolution of hidden states parameterized by Transformer-style dynamics.

Architecture:
    tokens → embeddings → Neural ODE Transformer → LM head → logits

Key features:
    - Continuous-depth processing via ODE integration
    - Discrete Transformer baseline for comparison
    - Full training pipeline with checkpointing
    - Text generation with multiple sampling strategies
    - GPU support via CUDA.jl
"""
module NeuralODELM

# Core modules
include("Config.jl")
include("Utils.jl")
include("Data.jl")
include("Training.jl")
include("Evaluation.jl")
include("Generation.jl")
include("Models/Models.jl")

# Re-export public API
using .Config: ConfigBundle, ModelConfig, TrainingConfig, DataConfig, load_config
using .Utils: select_device, set_seed
using .Data: build_tokenizer, load_corpus, encode_corpus, make_batches, 
             save_tokenizer, load_tokenizer, split_train_val
using .Training: train!, lm_loss, save_checkpoint, load_checkpoint
using .Evaluation: evaluate_perplexity, evaluate_loss
using .Generation: generate_text, generate_text_with_cache, sample_from_logits
using .Models: build_model

end # module

