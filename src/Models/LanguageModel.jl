"""
    LanguageModel.jl

End-to-end language model composition: embeddings → continuous/discrete transformer → LM head.
"""
module LanguageModel

using Flux
using DifferentialEquations
using ..Embeddings
using ..Attention
using ..ContinuousTransformer
using ..NeuralODEBlockModule
# ModelConfig will be available via parent module when this is included

export LanguageModelStruct, build_language_model

"""
    LanguageModelStruct

Top-level language model wrapper combining all components.
"""
struct LanguageModelStruct
    token_emb::TokenEmbedding
    pos_enc::PositionalEncoding
    core_block::Union{NeuralODEBlockModule.NeuralODEBlock, ContinuousTransformer.StackedTransformer}
    lm_head::Dense
    is_neural_ode::Bool
end

function (lm::LanguageModelStruct)(x)
    # x: (seq_len, batch)
    h = lm.token_emb(x)                     # (d_model, seq, batch)
    h = lm.pos_enc(h)
    h = lm.core_block(h)
    # reshape to (d_model, seq*batch) for lm_head
    d_model, seq, batch = size(h)
    h2 = reshape(h, d_model, seq * batch)
    logits = lm.lm_head(h2)                # (vocab_size, seq*batch)
    return reshape(logits, :, seq, batch)  # (vocab_size, seq, batch)
end

Flux.@functor LanguageModelStruct

"""
    build_language_model(mc)

Build either a discrete Transformer stack or a Neural ODE transformer
according to the configuration. mc should be a ModelConfig instance.
"""
function build_language_model(mc)
    token_emb = Embeddings.TokenEmbedding(mc.vocab_size, mc.d_model)
    pos_enc = Embeddings.PositionalEncoding(mc.d_model, mc.max_seq_len)

    # Select solver for Neural ODE
    solver = if mc.ode_solver == "Tsit5"
        "Tsit5"
    elseif mc.ode_solver == "RK4"
        "RK4"
    elseif mc.ode_solver == "Euler"
        "Euler"
    else
        "Tsit5"  # default
    end

    core_block = if mc.is_neural_ode
        NeuralODEBlockModule.NeuralODEBlock(mc.d_model, mc.n_heads, mc.d_ff;
                                      t0 = mc.ode_t0, t1 = mc.ode_t1,
                                      solver = solver,
                                      sensealg = mc.ode_sensealg,
                                      integrator_mode = mc.ode_integrator,
                                      nsteps = mc.ode_nsteps,
                                      reversible = mc.reversible,
                                      atol = mc.ode_atol,
                                      rtol = mc.ode_rtol)
    else
        ContinuousTransformer.StackedTransformer(mc.d_model, mc.n_heads, mc.d_ff; 
                                                 n_layers = mc.n_layers)
    end

    lm_head = Dense(mc.d_model, mc.vocab_size)

    return LanguageModelStruct(token_emb, pos_enc, core_block, lm_head, mc.is_neural_ode)
end

end # module
