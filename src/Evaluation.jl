"""
    Evaluation.jl

Evaluation metrics including perplexity and validation loss.
"""
module Evaluation

using ..Training: lm_loss

export evaluate_perplexity, evaluate_loss

"""
    evaluate_perplexity(model, batches, device_fn)

Compute perplexity (exp(mean_loss)) on a dataset.
"""
function evaluate_perplexity(model, batches, device_fn)
    total_loss = 0.0
    count = 0
    for (x, y) in batches
        x_d = device_fn(x)
        y_d = device_fn(y)
        loss = lm_loss(model, x_d, y_d)
        total_loss += loss
        count += 1
    end
    mean_loss = total_loss / max(count, 1)
    ppl = exp(mean_loss)
    return ppl, mean_loss
end

"""
    evaluate_loss(model, batches, device_fn)

Compute average loss on a dataset.
"""
function evaluate_loss(model, batches, device_fn)
    total_loss = 0.0
    count = 0
    for (x, y) in batches
        x_d = device_fn(x)
        y_d = device_fn(y)
        loss = lm_loss(model, x_d, y_d)
        total_loss += loss
        count += 1
    end
    return total_loss / max(count, 1)
end

end # module

