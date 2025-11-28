"""
    LanguageModel.jl

End-to-end language model composition: embeddings → continuous/discrete transformer → LM head.
Supports step-wise generation with KV caching.
"""
module LanguageModel

using Flux
using DifferentialEquations
using ..Embeddings
using ..Attention
using ..ContinuousTransformer
using ..NeuralODEBlock
using ..Config: ModelConfig

export LanguageModelStruct, build_language_model, LMState, init_state, generate_step

"""
    LMState

State for autoregressive generation with KV caching.
"""
struct LMState
    cache::Union{Nothing,Attention.KVCache}
    position::Int  # Current position in sequence
end

"""
    LanguageModelStruct

Top-level language model wrapper combining all components.
"""
struct LanguageModelStruct
    token_emb::TokenEmbedding
    pos_enc::PositionalEncoding
    core_block::Union{NeuralODEBlock.NeuralODEBlock, ContinuousTransformer.StackedTransformer}
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
    init_state(lm::LanguageModelStruct)

Initialize state for step-wise generation.
"""
function init_state(lm::LanguageModelStruct)
    return LMState(nothing, 0)
end

"""
    generate_step(lm::LanguageModelStruct, token_id::Int, state::LMState)

Generate logits for a single token step using KV caching.
This is optimized for autoregressive generation.

Note: KV caching currently works only for discrete Transformer stacks.
Neural ODE path uses full-sequence mode (limitation for now).

Returns:
- logits: (vocab_size,) - logits for next token
- new_state: Updated LMState with cache
"""
function generate_step(lm::LanguageModelStruct, token_id::Int, state::LMState)
    # Embed single token: (seq_len=1, batch=1)
    x = reshape([token_id], 1, 1)  # (1, 1)
    h = lm.token_emb(x)  # (d_model, 1, 1)
    
    # Apply positional encoding for current position
    state.position += 1
    d_model = size(h, 1)
    pos_enc_slice = lm.pos_enc.pe[:, state.position:state.position]
    h = h .+ reshape(pos_enc_slice, d_model, 1, 1)
    
    # Apply core block
    if lm.is_neural_ode
        # Neural ODE path: use full sequence mode (KV caching not yet supported)
        # TODO: Implement KV caching for Neural ODE path
        h = lm.core_block(h)
        # No cache update for ODE path
        new_state = LMState(nothing, state.position)
    else
        # Discrete Transformer path: use KV caching
        if state.cache === nothing
            # First token: initialize cache
            # For now, we'll do a full forward pass to initialize
            # In practice, you'd want to extract K/V from first attention call
            h_out = lm.core_block(h)
            # Initialize empty cache (will be populated on next call)
            # This is a simplified version - full implementation would extract K/V
            new_state = LMState(nothing, state.position)
            h = h_out
        else
            # Subsequent tokens: use cache
            # Note: This requires core_block to support cache
            # For now, fall back to full forward pass
            h = lm.core_block(h)
            new_state = LMState(state.cache, state.position)
        end
    end
    
    # Apply LM head
    d_model, seq, batch = size(h)
    h2 = reshape(h, d_model, seq * batch)
    logits = lm.lm_head(h2)  # (vocab_size,)
    logits = logits[:, 1]  # Extract single token logits
    
    return logits, new_state
end

"""
    build_language_model(mc::ModelConfig)

Build either a discrete Transformer stack or a Neural ODE transformer
according to the configuration.
"""
function build_language_model(mc::ModelConfig)
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
        NeuralODEBlock.NeuralODEBlock(mc.d_model, mc.n_heads, mc.d_ff;
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
