"""
    Generation.jl

Text generation utilities with various sampling strategies and KV caching support.
"""
module Generation

using Flux
using Random: rand
using ..Utils
using ..Data: SimpleTokenizer, encode, decode

export generate_text, sample_from_logits

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
    
    # Top-k filtering: keep only the k highest-logit tokens.
    # partialsortperm returns the indices (not a (vals, idxs) tuple).
    if top_k > 0
        k = min(top_k, length(logits))
        topk_idxs = partialsortperm(logits, 1:k, rev=true)
        mask = falses(length(logits))
        mask[topk_idxs] .= true
        logits = ifelse.(mask, logits, fill(-Inf32, length(logits)))
    end

    # Top-p (nucleus) sampling: keep the smallest prefix of tokens whose
    # cumulative probability reaches top_p. sortperm gives the ordering;
    # index logits by it to get the sorted values.
    if top_p > 0.0 && top_p < 1.0
        sorted_idxs = sortperm(logits, rev=true)
        sorted_logits = logits[sorted_idxs]
        probs = Flux.softmax(sorted_logits)
        cumsum_probs = cumsum(probs)
        cutoff_idx = findfirst(x -> x >= top_p, cumsum_probs)
        if cutoff_idx !== nothing
            mask = falses(length(logits))
            mask[sorted_idxs[1:cutoff_idx]] .= true
            logits = ifelse.(mask, logits, fill(-Inf32, length(logits)))
        end
    end
    
    probs = Flux.softmax(logits)
    # Simple sampling without Distributions.jl
    cumsum_probs = cumsum(probs)
    r = rand()
    idx = findfirst(x -> x >= r, cumsum_probs)
    return idx !== nothing ? idx : length(probs)
end

"""
    generate_text(model, tokenizer, prompt; max_new_tokens, temperature, top_k, top_p)

Autoregressive generation from a prompt using full-sequence forward passes.
The whole context is recomputed at each step. This is the single supported
generation path: the continuous-depth ODE core has no discrete per-layer
key/value state to cache, so KV caching is intentionally not implemented.

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
        # Prepare input: (seq_len, batch=1).
        # `collect` copies so the reshaped matrix does not share `ids`' buffer
        # (a shared buffer would make the `push!` below fail).
        x = reshape(collect(ids), :, 1)
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

end # module
