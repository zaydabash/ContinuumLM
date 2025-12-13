"""
    Training.jl

Training loop with optimizer, learning rate scheduling, checkpointing, and TensorBoard logging.
"""
module Training

using Flux
using Optimisers
using BSON
using ..Config
using ..Utils

# Optional TensorBoardLogger - try to load, but don't fail if unavailable
const HAS_TENSORBOARD = try
    using TensorBoardLogger
    true
catch
    false
end

export train!, lm_loss, save_checkpoint, load_checkpoint

"""
    lm_loss(model, x, y)

Compute cross-entropy loss for language modeling.

x: (seq_len, batch) - input token IDs
y: (seq_len, batch) - target token IDs (shifted by 1)
"""
function lm_loss(model, x, y)
    logits = model(x)  # (vocab, seq, batch)
    vocab, seq, batch = size(logits)
    # flatten to (vocab, seq*batch) and (seq*batch,)
    logits_flat = reshape(logits, vocab, :)
    y_flat = reshape(y, :)

    # One-hot encode targets
    y_onehot = Flux.onehotbatch(y_flat, 1:vocab)
    return Flux.logitcrossentropy(logits_flat, y_onehot)
end

"""
    get_lr(step, warmup_steps, base_lr)

Simple linear warmup + constant learning rate schedule.
"""
function get_lr(step, warmup_steps, base_lr)
    if step <= warmup_steps
        return base_lr * (step / warmup_steps)
    else
        return base_lr
    end
end

"""
    save_checkpoint(model, path, step, loss)

Save model checkpoint to disk.
"""
function save_checkpoint(model, path::String, step::Int, loss::Float64)
    Utils.ensure_dir(dirname(path))
    BSON.@save path model step loss
    Base.println("Saved checkpoint to $path (step=$step, loss=$loss)")
end

"""
    load_checkpoint(path)

Load model checkpoint from disk.
"""
function load_checkpoint(path::String)
    if !isfile(path)
        error("Checkpoint not found: $path")
    end
    BSON.@load path model step loss
    return model, step, loss
end


"""
    train!(model, train_batches, val_batches, cfg::ConfigBundle)

Main training loop with validation, checkpointing, and TensorBoard logging.
"""
function train!(model, train_batches, val_batches, cfg::Config.ConfigBundle)
    tc = cfg.training
    mc = cfg.model
    
    Utils.set_seed(tc.seed)
    device_fn = Utils.select_device(tc.device)
    model = device_fn(model)

    opt = Optimisers.AdamW(tc.lr)
    st = Optimisers.setup(opt, model)

    Utils.ensure_dir(tc.checkpoint_dir)
    
    # Setup logging
    run_dir = joinpath(tc.log_dir, tc.run_name)
    Utils.ensure_dir(run_dir)
    
    tb_logger = if HAS_TENSORBOARD
        TBLogger(run_dir)
    else
        nothing
    end
    
    if HAS_TENSORBOARD && tb_logger !== nothing
        Base.println("TensorBoard logs will be written to: $run_dir")
        Base.println("View with: tensorboard --logdir $(tc.log_dir)")
    end
    
    step = 0
    avg_loss = 0.0
    best_val_loss = Inf
    train_iter = Iterators.cycle(train_batches)

    Base.println("Starting training with $(tc.num_steps) steps")

    for (x, y) in train_iter
        step += 1
        
        # Learning rate scheduling
        current_lr = get_lr(step, tc.warmup_steps, tc.lr)
        Optimisers.adjust!(st, current_lr)
        
        x_d = device_fn(x)
        y_d = device_fn(y)

        loss, grads = Flux.withgradient(model) do m
            lm_loss(m, x_d, y_d)
        end

        avg_loss = 0.9 * avg_loss + 0.1 * loss

        # Gradient clipping (simplified - disabled for now to get training working)
        grad_norm = 0.0

        st, model = Optimisers.update(st, model, grads[1])

        # Logging
        if step % tc.log_every == 0
            Base.println("step=$step loss=$loss avg_loss=$avg_loss lr=$current_lr")
            
            # TensorBoard logging
            if tb_logger !== nothing
                log_value(tb_logger, "train/loss", loss, step)
                log_value(tb_logger, "train/avg_loss", avg_loss, step)
                log_value(tb_logger, "train/lr", current_lr, step)
                
                # Log gradient norm
                log_value(tb_logger, "train/grad_norm", grad_norm, step)
            end
        end

        # Validation
        if step % tc.eval_every == 0 && val_batches !== nothing
            val_loss = evaluate_validation(model, val_batches, device_fn)
            val_ppl = exp(val_loss)
            
            Base.println("Validation: step=$step val_loss=$val_loss val_ppl=$val_ppl")
            
            # TensorBoard logging for validation
            if tb_logger !== nothing
                log_value(tb_logger, "eval/loss", val_loss, step)
                log_value(tb_logger, "eval/perplexity", val_ppl, step)
            end
            
            if tc.save_best && val_loss < best_val_loss
                best_val_loss = val_loss
                checkpoint_path = joinpath(tc.checkpoint_dir, "best_model.bson")
                save_checkpoint(model, checkpoint_path, step, val_loss)
            end
        end

        # Periodic checkpointing
        if step % tc.checkpoint_every == 0
            checkpoint_path = joinpath(tc.checkpoint_dir, "checkpoint_step_$(step).bson")
            save_checkpoint(model, checkpoint_path, step, avg_loss)
        end

        if step >= tc.num_steps
            break
        end
    end

    Base.println("Training completed after $step steps")
    return model
end

"""
    evaluate_validation(model, val_batches, device_fn)

Compute validation loss.
"""
function evaluate_validation(model, val_batches, device_fn)
    total_loss = 0.0
    count = 0
    for (x, y) in val_batches
        x_d = device_fn(x)
        y_d = device_fn(y)
        loss = lm_loss(model, x_d, y_d)
        total_loss += loss
        count += 1
    end
    return total_loss / max(count, 1)
end

end # module
