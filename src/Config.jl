"""
    Config.jl

Configuration management for Neural ODE Language Model.
Provides typed config structs and TOML loading functionality.
"""
module Config

using TOML

"""
    ModelConfig

Configuration for the language model architecture.
"""
Base.@kwdef struct ModelConfig
    d_model::Int = 256
    n_heads::Int = 4
    d_ff::Int = 1024
    vocab_size::Int = 8000
    max_seq_len::Int = 128
    is_neural_ode::Bool = true
    ode_t0::Float64 = 0.0
    ode_t1::Float64 = 1.0
    ode_solver::String = "Tsit5"
    ode_sensealg::String = "InterpolatingAdjoint"  # Adjoint sensitivity method
    ode_atol::Float64 = 1e-6  # Absolute tolerance for ODE solver
    ode_rtol::Float64 = 1e-6  # Relative tolerance for ODE solver
    ode_integrator::String = "generic"  # "generic" or "custom_fixed_step"
    ode_nsteps::Int = 4  # Number of steps for custom fixed-step integrator
    reversible::Bool = false  # Use reversible ODE for memory efficiency
    n_layers::Int = 4  # for discrete transformer baseline
end

"""
    TrainingConfig

Configuration for training.
"""
Base.@kwdef struct TrainingConfig
    batch_size::Int = 16
    seq_len::Int = 128
    num_steps::Int = 10_000
    log_every::Int = 100
    eval_every::Int = 1000
    lr::Float64 = 1e-3
    weight_decay::Float64 = 0.01
    grad_clip::Float64 = 1.0
    warmup_steps::Int = 500
    device::String = "auto"  # "cpu", "gpu", or "auto"
    checkpoint_dir::String = "checkpoints"
    checkpoint_every::Int = 1000
    save_best::Bool = true
    seed::Int = 42
    log_dir::String = "logs"  # Directory for TensorBoard logs
    run_name::String = "default_run"  # Name for this training run
end

"""
    DataConfig

Configuration for data and tokenization.
"""
Base.@kwdef struct DataConfig
    corpus_path::String = "data/corpus.txt"
    tokenizer_path::String = "data/tokenizer.json"
    vocab_size::Int = 8000
    train_split::Float64 = 0.9
end

"""
    ConfigBundle

Top-level configuration combining model, training, and data.
"""
Base.@kwdef struct ConfigBundle
    model::ModelConfig = ModelConfig()
    training::TrainingConfig = TrainingConfig()
    data::DataConfig = DataConfig()
end

"""
    load_config(path::String) -> ConfigBundle

Load configuration from a TOML file and return a ConfigBundle.
"""
function load_config(path::String)
    d = TOML.parsefile(path)

    m = get(d, "model", Dict())
    t = get(d, "training", Dict())
    dat = get(d, "data", Dict())

    mc = ModelConfig(
        d_model = get(m, "d_model", 256),
        n_heads = get(m, "n_heads", 4),
        d_ff = get(m, "d_ff", 1024),
        vocab_size = get(m, "vocab_size", 8000),
        max_seq_len = get(m, "max_seq_len", 128),
        is_neural_ode = get(m, "is_neural_ode", true),
        ode_t0 = get(m, "ode_t0", 0.0),
        ode_t1 = get(m, "ode_t1", 1.0),
        ode_solver = get(m, "ode_solver", "Tsit5"),
        ode_sensealg = get(m, "ode_sensealg", "InterpolatingAdjoint"),
        ode_atol = get(m, "ode_atol", 1e-6),
        ode_rtol = get(m, "ode_rtol", 1e-6),
        ode_integrator = get(m, "ode_integrator", "generic"),
        ode_nsteps = get(m, "ode_nsteps", 4),
        reversible = get(m, "reversible", false),
        n_layers = get(m, "n_layers", 4),
    )

    tc = TrainingConfig(
        batch_size = get(t, "batch_size", 16),
        seq_len = get(t, "seq_len", 128),
        num_steps = get(t, "num_steps", 10_000),
        log_every = get(t, "log_every", 100),
        eval_every = get(t, "eval_every", 1000),
        lr = get(t, "lr", 1e-3),
        weight_decay = get(t, "weight_decay", 0.01),
        grad_clip = get(t, "grad_clip", 1.0),
        warmup_steps = get(t, "warmup_steps", 500),
        device = get(t, "device", "auto"),
        checkpoint_dir = get(t, "checkpoint_dir", "checkpoints"),
        checkpoint_every = get(t, "checkpoint_every", 1000),
        save_best = get(t, "save_best", true),
        seed = get(t, "seed", 42),
        log_dir = get(t, "log_dir", "logs"),
        run_name = get(t, "run_name", "default_run"),
    )

    dc = DataConfig(
        corpus_path = get(dat, "corpus_path", "data/corpus.txt"),
        tokenizer_path = get(dat, "tokenizer_path", "data/tokenizer.json"),
        vocab_size = get(dat, "vocab_size", 8000),
        train_split = get(dat, "train_split", 0.9),
    )

    return ConfigBundle(model = mc, training = tc, data = dc)
end

end # module
