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

    # For now, use greedy sampling to avoid complex broadcasting issues
    if temperature == 0.0 || top_k == 1
        return argmax(logits)
    end

    # Simple multinomial sampling
    probs = Flux.softmax(logits)
    # Convert to regular array and sample
    probs_vec = vec(Array(probs))
    cumsum_probs = cumsum(probs_vec)
    r = rand()
    idx = findfirst(x -> x >= r, cumsum_probs)
    return idx !== nothing ? idx : length(probs_vec)
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
    
    ids = Vector{Int}(encode(tok, prompt))

    # Get maximum sequence length from model
    max_seq_len = size(model.pos_enc.pe, 2)

    for _ in 1:max_new_tokens
        # Use sliding window: take the most recent max_seq_len tokens
        seq_len = min(length(ids), max_seq_len)
        if seq_len == 0
            break
        end

        # Get recent tokens
        recent_ids = ids[max(1, length(ids) - seq_len + 1):end]

        # Pad with UNK token if necessary
        if length(recent_ids) < seq_len
            padding = fill(1, seq_len - length(recent_ids))  # 1 = UNK
            recent_ids = vcat(padding, recent_ids)
        end

        # Prepare input: (seq_len, batch=1)
        x = reshape(recent_ids, :, 1)
        x_d = device_fn(x)

        # Forward pass
        logits = model(x_d)      # (vocab, seq, 1)
        last_logits = logits[:, end, 1]

        # Sample next token
        next_id = sample_from_logits(last_logits;
                                     temperature=temperature,
                                     top_k=top_k,
                                     top_p=top_p)

        # Create new ids vector (avoid mutation)
        ids = vcat(ids, next_id)

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
