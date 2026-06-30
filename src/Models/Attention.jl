"""
    Attention.jl

Multi-head self-attention and Transformer block components.
"""
module Attention

using Flux
using NNlib
using ChainRulesCore: @ignore_derivatives

export MultiHeadSelfAttention, FeedForwardBlock, TransformerBlock

"""
    MultiHeadSelfAttention

Multi-head self-attention mechanism with causal masking.
"""
struct MultiHeadSelfAttention
    Wq::Dense
    Wk::Dense
    Wv::Dense
    Wo::Dense
    n_heads::Int
    d_head::Int
end

function MultiHeadSelfAttention(d_model::Int, n_heads::Int)
    @assert d_model % n_heads == 0 "d_model must be divisible by n_heads"
    d_head = div(d_model, n_heads)
    Wq = Dense(d_model, d_model)
    Wk = Dense(d_model, d_model)
    Wv = Dense(d_model, d_model)
    Wo = Dense(d_model, d_model)
    return MultiHeadSelfAttention(Wq, Wk, Wv, Wo, n_heads, d_head)
end

function (m::MultiHeadSelfAttention)(x; mask::Bool=true)
    # x: (d_model, seq_len, batch)
    d_model, seq_len, batch = size(x)

    # Full-sequence attention via batched matmul, no in-place mutation,
    # fully Zygote-compatible.
    Q = m.Wq(x)  # (d_model, seq, batch)
    K = m.Wk(x)
    V = m.Wv(x)

    # Reshape and merge heads into batch dim for batched matmul
    # (d_model, seq, batch) -> (d_head, n_heads, seq, batch)
    #                       -> (d_head, seq, n_heads*batch) via permute+reshape
    Q_r = reshape(Q, m.d_head, m.n_heads, seq_len, batch)
    K_r = reshape(K, m.d_head, m.n_heads, seq_len, batch)
    V_r = reshape(V, m.d_head, m.n_heads, seq_len, batch)

    Q_p = reshape(permutedims(Q_r, (1, 3, 2, 4)), m.d_head, seq_len, m.n_heads * batch)
    K_p = reshape(permutedims(K_r, (1, 3, 2, 4)), m.d_head, seq_len, m.n_heads * batch)
    V_p = reshape(permutedims(V_r, (1, 3, 2, 4)), m.d_head, seq_len, m.n_heads * batch)

    scale = 1f0 / sqrt(Float32(m.d_head))

    # scores: (seq, seq, n_heads*batch)
    # K^T @ Q: (seq, d_head, nb) @ (d_head, seq, nb) = (seq, seq, nb)
    scores = NNlib.batched_mul(permutedims(K_p, (2, 1, 3)), Q_p) .* scale

    if mask
        # scores is indexed [key_pos, query_pos, ...] (see batched_mul above: the
        # first operand's seq axis -> key, Q_p's seq axis -> query). A query may
        # only attend to keys at or before it, so mask out key_pos > query_pos.
        # @ignore_derivatives tells Zygote the mask has no learnable params
        causal = @ignore_derivatives Float32[key_pos > query_pos ? -1f6 : 0f0 for key_pos in 1:seq_len, query_pos in 1:seq_len]
        scores = scores .+ causal
    end

    attn = Flux.softmax(scores, dims=1)  # (seq, seq, n_heads*batch)

    # Z: V @ attn -> (d_head, seq, nb) @ (seq, seq, nb) = (d_head, seq, nb)
    Z_p = NNlib.batched_mul(V_p, attn)

    # Reshape back: (d_head, seq, n_heads*batch) -> (d_model, seq, batch)
    Z_r = reshape(Z_p, m.d_head, seq_len, m.n_heads, batch)
    Z = reshape(permutedims(Z_r, (1, 3, 2, 4)), d_model, seq_len, batch)

    return m.Wo(Z)
end

Flux.@functor MultiHeadSelfAttention

"""
    FeedForwardBlock

Two-layer MLP with activation function.
"""
struct FeedForwardBlock
    proj1::Dense
    proj2::Dense
    activation
end

function FeedForwardBlock(d_model::Int, d_ff::Int; activation=Flux.gelu)
    proj1 = Dense(d_model, d_ff, activation)
    proj2 = Dense(d_ff, d_model)
    return FeedForwardBlock(proj1, proj2, activation)
end

(ff::FeedForwardBlock)(x) = ff.proj2(ff.proj1(x))

Flux.@functor FeedForwardBlock

"""
    TransformerBlock

Complete Transformer block with self-attention, feedforward, and layer norms.
"""
struct TransformerBlock
    attn::MultiHeadSelfAttention
    ff::FeedForwardBlock
    norm1::LayerNorm
    norm2::LayerNorm
end

function TransformerBlock(d_model::Int, n_heads::Int, d_ff::Int)
    attn = MultiHeadSelfAttention(d_model, n_heads)
    ff = FeedForwardBlock(d_model, d_ff)
    norm1 = LayerNorm(d_model)
    norm2 = LayerNorm(d_model)
    return TransformerBlock(attn, ff, norm1, norm2)
end

function (tb::TransformerBlock)(x; mask::Bool=true)
    # x: (d_model, seq, batch), Post-LN
    h = tb.norm1(x .+ tb.attn(x; mask=mask))
    h2 = tb.norm2(h .+ tb.ff(h))
    return h2
end

Flux.@functor TransformerBlock

end # module
