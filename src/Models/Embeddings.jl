"""
    Embeddings.jl

Token and positional embedding layers.
"""
module Embeddings

using Flux

export TokenEmbedding, PositionalEncoding

"""
    TokenEmbedding

Standard token embedding layer mapping token IDs to hidden dimension.
"""
struct TokenEmbedding
    emb::Flux.Embedding
end

TokenEmbedding(vocab_size::Int, d_model::Int) = TokenEmbedding(Flux.Embedding(vocab_size, d_model))

function (te::TokenEmbedding)(x)
    # x: (seq_len, batch)
    return te.emb(x)
end


"""
    PositionalEncoding

Sinusoidal positional encoding for sequences.
"""
struct PositionalEncoding
    pe::Array{Float32, 2}  # (d_model, max_seq_len)
end

"""
    PositionalEncoding(d_model, max_seq_len)

Standard sinusoidal positional encoding.
"""
function PositionalEncoding(d_model::Int, max_seq_len::Int)
    pe = zeros(Float32, d_model, max_seq_len)
    for pos in 0:max_seq_len-1
        for i in 0:(d_model÷2 - 1)
            denom = 10000.0^(2i/d_model)
            pe[2i+1, pos+1] = sin(pos/denom)
            if 2i+2 <= d_model
                pe[2i+2, pos+1] = cos(pos/denom)
            end
        end
    end
    return PositionalEncoding(pe)
end

function (p::PositionalEncoding)(x)
    # x: (d_model, seq_len, batch)
    d_model, seq_len, batch = size(x)
    if seq_len > size(p.pe, 2)
        error("Sequence length $seq_len exceeds max_seq_len $(size(p.pe, 2))")
    end
    pe_slice = p.pe[:, 1:seq_len]
    # broadcast add positional encodings across batch
    return x .+ reshape(pe_slice, d_model, seq_len, 1)
end


end # module

