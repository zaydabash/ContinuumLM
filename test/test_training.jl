"""
    test_training.jl

Tests for training loop, loss computation, and checkpointing.
"""

using Test
using NeuralODELM
using NeuralODELM.Config
using NeuralODELM.Data
using NeuralODELM.Models
using NeuralODELM.Training
using NeuralODELM.Utils

function test_training()
    # Create a tiny test corpus
    test_corpus = """
    The cat sat on the mat.
    The dog ran in the park.
    The bird flew in the sky.
    """ |> strip
    
    # Build tokenizer and encode
    tok = build_tokenizer(test_corpus; vocab_size=50)
    seqs = encode_corpus(tok, test_corpus; seq_len=8)
    
    # Split into train/val
    train_seqs, val_seqs = split_train_val(seqs, 0.7)
    train_batches = make_batches(train_seqs, 2)
    val_batches = length(val_seqs) > 0 ? make_batches(val_seqs, 2) : nothing
    
    # Create tiny model config
    mc = ModelConfig(
        d_model=32,
        n_heads=2,
        d_ff=64,
        vocab_size=50,
        max_seq_len=8,
        is_neural_ode=false,  # Use discrete for faster tests
        n_layers=1
    )
    
    tc = TrainingConfig(
        batch_size=2,
        seq_len=8,
        num_steps=10,  # Very few steps for test
        log_every=5,
        eval_every=10,
        lr=1e-3,
        weight_decay=0.01,
        grad_clip=1.0,
        warmup_steps=5,
        device="cpu",
        checkpoint_dir="test_checkpoints",
        checkpoint_every=10,
        save_best=false,
        seed=42
    )
    
    dc = DataConfig(
        corpus_path="",
        tokenizer_path="",
        vocab_size=50,
        train_split=0.7
    )
    
    cfg = ConfigBundle(model=mc, training=tc, data=dc)
    
    # Build model
    model = Models.build_model(mc)
    
    # Test loss computation
    x = rand(1:50, 7, 2)
    y = rand(1:50, 7, 2)
    loss = lm_loss(model, x, y)
    @test !isnan(loss)
    @test loss >= 0
    
    # Test training (very short run)
    set_seed(42)
    trained_model = train!(model, train_batches, val_batches, cfg)
    @test trained_model isa Models.LanguageModel.LanguageModelStruct
    
    # Test checkpoint saving/loading
    checkpoint_path = "test_checkpoints/test_model.bson"
    save_checkpoint(model, checkpoint_path, 1, Float32(0.5))
    @test isfile(checkpoint_path)
    
    loaded_model, step, loss_val = load_checkpoint(checkpoint_path)
    @test step == 1
    @test loss_val == 0.5f0
    
    # Cleanup
    if isfile(checkpoint_path)
        rm(checkpoint_path)
    end
    if isdir("test_checkpoints")
        rm("test_checkpoints", recursive=true)
    end
end

