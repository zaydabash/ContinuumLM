"""
    ContinuousTransformer.jl

Discrete Transformer stack (baseline) and continuous transformer interface.
"""
module ContinuousTransformer

using Flux
using ..Attention: TransformerBlock

export StackedTransformer

"""
    StackedTransformer

Discrete stack of Transformer blocks (baseline model).
"""
struct StackedTransformer
    blocks::Vector{TransformerBlock}
end

function StackedTransformer(d_model::Int, n_heads::Int, d_ff::Int; n_layers::Int=4)
    blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in 1:n_layers]
    return StackedTransformer(blocks)
end

function (st::StackedTransformer)(x)
    h = x
    for blk in st.blocks
        h = blk(h; mask=true)
    end
    return h
end


end # module

