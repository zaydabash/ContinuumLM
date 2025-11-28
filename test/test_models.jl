"""
    test_models.jl

Tests for model components: embeddings, attention, Neural ODE block, and full model.
"""

using Test
using Flux
using NeuralODELM
using NeuralODELM.Models
using NeuralODELM.Config

function test_models()
    d_model = 64
    vocab_size = 100
    seq_len = 16
    batch_size = 2
    
    # Test token embedding
    token_emb = Models.Embeddings.TokenEmbedding(vocab_size, d_model)
    x_tokens = rand(1:vocab_size, seq_len, batch_size)
    h = token_emb(x_tokens)
    @test size(h) == (d_model, seq_len, batch_size)
    
    # Test positional encoding
    pos_enc = Models.Embeddings.PositionalEncoding(d_model, seq_len)
    h_pos = pos_enc(h)
    @test size(h_pos) == (d_model, seq_len, batch_size)
    
    # Test attention block
    attn = Models.Attention.MultiHeadSelfAttention(d_model, 4)
    h_attn = attn(h; mask=true)
    @test size(h_attn) == (d_model, seq_len, batch_size)
    
    # Test feedforward block
    ff = Models.Attention.FeedForwardBlock(d_model, 256)
    h_ff = ff(h)
    @test size(h_ff) == (d_model, seq_len, batch_size)
    
    # Test Transformer block
    tb = Models.Attention.TransformerBlock(d_model, 4, 256)
    h_tb = tb(h; mask=true)
    @test size(h_tb) == (d_model, seq_len, batch_size)
    
    # Test discrete Transformer stack
    stacked = Models.ContinuousTransformer.StackedTransformer(d_model, 4, 256; n_layers=2)
    h_stacked = stacked(h)
    @test size(h_stacked) == (d_model, seq_len, batch_size)
    
    # Test Neural ODE block (smaller for speed)
    # Note: This may be slow, so we use a very small configuration
    ode_block = Models.NeuralODEBlock.NeuralODEBlock(d_model, 2, 128; t0=0.0, t1=0.5)
    h_ode = ode_block(h)
    @test size(h_ode) == (d_model, seq_len, batch_size)
    
    # Test full language model (discrete)
    mc_discrete = ModelConfig(
        d_model=d_model,
        n_heads=2,
        d_ff=256,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        is_neural_ode=false,
        n_layers=2
    )
    model_discrete = Models.build_model(mc_discrete)
    logits_discrete = model_discrete(x_tokens)
    @test size(logits_discrete) == (vocab_size, seq_len, batch_size)
    
    # Test full language model (Neural ODE) - use shorter integration time for speed
    mc_ode = ModelConfig(
        d_model=d_model,
        n_heads=2,
        d_ff=256,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        is_neural_ode=true,
        ode_t1=0.5,  # Shorter integration for faster tests
        n_layers=2
    )
    model_ode = Models.build_model(mc_ode)
    logits_ode = model_ode(x_tokens)
    @test size(logits_ode) == (vocab_size, seq_len, batch_size)
    
    # Test gradient flow
    loss_discrete, back_discrete = Flux.withgradient(() -> 
        sum(model_discrete(x_tokens)), Flux.params(model_discrete))
    @test !isnan(loss_discrete)
    
    # Note: Neural ODE gradients can be expensive, so we skip in basic tests
    # but the structure should work
end

