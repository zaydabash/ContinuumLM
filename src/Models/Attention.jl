"""
    Attention.jl

Multi-head self-attention and Transformer block components with KV caching support.
"""
module Attention

using Flux
using NNlib
using ChainRulesCore: @ignore_derivatives

export MultiHeadSelfAttention, FeedForwardBlock, TransformerBlock, KVCache

"""
    KVCache

Key-Value cache for efficient autoregressive generation.
Stores cached keys and values from previous tokens to avoid recomputation.

Fields:
- K: Cached keys (n_heads, d_head, seq_len, batch)
- V: Cached values (n_heads, d_head, seq_len, batch)
- seq_len: Current sequence length in cache
"""
struct KVCache
    K::Array{Float32,4}  # (n_heads, d_head, seq, batch)
    V::Array{Float32,4}  # (n_heads, d_head, seq, batch)
    seq_len::Int
end

"""
    MultiHeadSelfAttention

Multi-head self-attention mechanism with causal masking and KV caching support.
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

function (m::MultiHeadSelfAttention)(x; mask::Bool=true, cache::Union{Nothing,KVCache}=nothing)
    # x: (d_model, seq_len, batch)
    d_model, seq_len, batch = size(x)
    
    # For generation with cache: only compute Q/K/V for new tokens
    if cache !== nothing
        # Generation mode: seq_len should be 1 for new token
        @assert seq_len == 1 && batch == 1 "Cache mode requires seq_len=1, batch=1"
        
        Q = m.Wq(x)  # (d_model, 1, 1)
        K_new = m.Wk(x)  # (d_model, 1, 1)
        V_new = m.Wv(x)  # (d_model, 1, 1)
        
        # Split heads
        Qh = reshape(Q, m.n_heads, m.d_head, 1, 1)
        Kh_new = reshape(K_new, m.n_heads, m.d_head, 1, 1)
        Vh_new = reshape(V_new, m.n_heads, m.d_head, 1, 1)
        
        # Append to cache
        K_cached = cat(cache.K, Kh_new, dims=3)  # (n_heads, d_head, seq_len+1, 1)
        V_cached = cat(cache.V, Vh_new, dims=3)
        
        # Attention: Q (new token) attends to all cached K/V
        Qhb = Qh[:, :, 1, 1]  # (d_head,)
        Khb_all = K_cached[:, :, :, 1]  # (n_heads, d_head, total_seq)
        Vhb_all = V_cached[:, :, :, 1]  # (n_heads, d_head, total_seq)
        
        # Scaled dot-product attention
        scale = 1f0 / sqrt(Float32(m.d_head))
        scores = (Qhb' * Khb_all) .* scale  # (total_seq,)
        
        # Causal mask: new token can attend to all previous tokens
        attn = Flux.softmax(scores, dims=1)
        Zh_b = Vhb_all * attn  # (d_head,)
        
        # Combine heads: (d_head,) -> (d_model, 1, 1)
        Zh = reshape(Zh_b, m.d_head, 1, 1)
        Z = reshape(Zh, d_model, 1, 1)
        
        # Update cache
        new_cache = KVCache(K_cached, V_cached, cache.seq_len + 1)
        
        return m.Wo(Z), new_cache
    else
        # Training mode: full sequence, no cache
        # Use batched matmul — no in-place mutation, fully Zygote-compatible
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
            # causal mask: position i cannot attend to position j > i
            # @ignore_derivatives tells Zygote the mask has no learnable params
            causal = @ignore_derivatives Float32[j > i ? -1f6 : 0f0 for i in 1:seq_len, j in 1:seq_len]
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
Supports KV caching for generation.
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

function (tb::TransformerBlock)(x; mask::Bool=true, cache::Union{Nothing,KVCache}=nothing)
    # x: (d_model, seq, batch)
    if cache !== nothing
        # Generation mode with cache — Post-LN matching training mode exactly
        h_attn, new_cache = tb.attn(x; mask=mask, cache=cache)
        h = tb.norm1(x .+ h_attn)
        h2 = tb.norm2(h .+ tb.ff(h))
        return h2, new_cache
    else
        # Training mode: Post-LN
        h = tb.norm1(x .+ tb.attn(x; mask=mask))
        h2 = tb.norm2(h .+ tb.ff(h))
        return h2
    end
end

Flux.@functor TransformerBlock

end # module
