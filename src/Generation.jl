"""
    Generation.jl

Text generation utilities with various sampling strategies and KV caching support.
"""
module Generation

using Flux
using ..Utils
using ..Data: SimpleTokenizer, encode, decode
using ..Models

export generate_text, sample_from_logits, generate_text_with_cache

"""
    sample_from_logits(logits; temperature, top_k, top_p)

Sample a token from logits using temperature, top-k, or top-p (nucleus) sampling.

logits: (vocab,) - unnormalized log probabilities
"""
function sample_from_logits(logits; temperature=1.0, top_k=0, top_p=0.0)
    # logits: (vocab,)
    if temperature != 1.0
        logits = logits ./ temperature
    end
    
    # Top-k filtering
    if top_k > 0
        k = min(top_k, length(logits))
        topk_vals, topk_idxs = partialsortperm(logits, rev=true, 1:k)
        mask = falses(length(logits))
        mask[topk_idxs] .= true
        logits = ifelse.(mask, logits, fill(-Inf, length(logits)))
    end
    
    # Top-p (nucleus) sampling
    if top_p > 0.0 && top_p < 1.0
        sorted_logits, sorted_idxs = sort(logits, rev=true)
        probs = softmax(sorted_logits)
        cumsum_probs = cumsum(probs)
        cutoff_idx = findfirst(x -> x >= top_p, cumsum_probs)
        if cutoff_idx !== nothing
            mask = falses(length(logits))
            mask[sorted_idxs[1:cutoff_idx]] .= true
            logits = ifelse.(mask, logits, fill(-Inf, length(logits)))
        end
    end
    
    probs = softmax(logits)
    # Simple sampling without Distributions.jl
    cumsum_probs = cumsum(probs)
    r = Base.Random.rand()
    idx = findfirst(x -> x >= r, cumsum_probs)
    return idx !== nothing ? idx : length(probs)
end

"""
    generate_text(model, tokenizer, prompt; max_new_tokens, temperature, top_k, top_p)

Simple autoregressive generation from a prompt (legacy full-sequence mode).
This recomputes the entire sequence at each step.

Returns generated text as a string.
"""
function generate_text(model, tok::SimpleTokenizer, prompt::String;
                       max_new_tokens=100,
                       temperature=1.0,
                       top_k=0,
                       top_p=0.0)
    device_fn = Utils.select_device("auto")
    model = device_fn(model)
    
    ids = encode(tok, prompt)
    
    for _ in 1:max_new_tokens
        # Prepare input: (seq_len, batch=1)
        x = reshape(ids, :, 1)
        x_d = device_fn(x)
        
        # Forward pass
        logits = model(x_d)      # (vocab, seq, 1)
        last_logits = logits[:, end, 1]
        
        # Sample next token
        next_id = sample_from_logits(last_logits; 
                                     temperature=temperature, 
                                     top_k=top_k, 
                                     top_p=top_p)
        push!(ids, next_id)
        
        # Optional: stop at EOS token if tokenizer has one
        # This is a simplified version - real tokenizers have special tokens
    end
    
    return decode(tok, ids)
end

"""
    generate_text_with_cache(model, tokenizer, prompt; max_new_tokens, temperature, top_k, top_p)

Efficient autoregressive generation using KV caching.
This uses step-wise generation with cached keys/values for faster inference.

Returns generated text as a string.
"""
function generate_text_with_cache(model, tok::SimpleTokenizer, prompt::String;
                                  max_new_tokens=100,
                                  temperature=1.0,
                                  top_k=0,
                                  top_p=0.0)
    device_fn = Utils.select_device("auto")
    model = device_fn(model)
    
    # Encode prompt
    ids = encode(tok, prompt)
    
    # Initialize state for step-wise generation
    state = Models.LanguageModel.init_state(model)
    
    # Process initial prompt tokens (if any) to initialize cache
    # For now, we'll process them one by one
    for token_id in ids
        logits, state = Models.LanguageModel.generate_step(model, token_id, state)
    end
    
    # Generate new tokens
    for _ in 1:max_new_tokens
        # Get logits for current state
        last_token_id = ids[end]
        logits, state = Models.LanguageModel.generate_step(model, last_token_id, state)
        
        # Sample next token
        next_id = sample_from_logits(logits; 
                                     temperature=temperature, 
                                     top_k=top_k, 
                                     top_p=top_p)
        push!(ids, next_id)
        
        # Update state with new token
        state = Models.LanguageModel.LMState(state.cache, state.position)
    end
    
    return decode(tok, ids)
end

end # module
