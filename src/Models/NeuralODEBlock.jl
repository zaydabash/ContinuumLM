"""
    NeuralODEBlock.jl

Neural ODE block implementing continuous-depth Transformer with advanced features:
- Proper adjoint sensitivity methods for efficient backpropagation
- Custom continuous-attention kernel integrator (RK4-style)
- Reversible ODE design for memory efficiency

The hidden state evolves continuously via an ODE parameterized by Transformer dynamics.
"""
module NeuralODEBlockModule

using Flux
using Functors
using DifferentialEquations
using DiffEqFlux
using DiffEqFlux: InterpolatingAdjoint, BacksolveAdjoint, QuadratureAdjoint, ZygoteVJP
using ChainRulesCore: @ignore_derivatives
using ..Attention: TransformerBlock

"""
    NeuralODEBlock

Wrap a TransformerBlock into a Neural ODE over continuous depth.
The ODE integrates dh/dt = f(h, t, θ) where f is the Transformer block.

Fields:
- block: Transformer block that defines the dynamics
- tspan: Integration time interval (t0, t1)
- solver: ODE solver (Tsit5, RK4, etc.)
- sensealg: Adjoint sensitivity algorithm for backpropagation
- integrator_mode: :generic (use DifferentialEquations) or :custom_fixed_step (use custom RK4)
- nsteps: Number of steps for custom fixed-step integrator
- reversible: Whether to use reversible ODE for memory efficiency
- atol: Absolute tolerance for ODE solver
- rtol: Relative tolerance for ODE solver
"""
struct NeuralODEBlock
    block::TransformerBlock
    tspan::Tuple{Float64,Float64}
    solver
    sensealg
    integrator_mode::Symbol
    nsteps::Int
    reversible::Bool
    atol::Float64
    rtol::Float64
end

"""
    NeuralODEBlock(d_model, n_heads, d_ff; kwargs...)

Construct a NeuralODEBlock with configurable ODE integration and adjoint methods.

Keyword arguments:
- t0, t1: Integration time interval (default: 0.0, 1.0)
- solver: ODE solver string or instance (default: "Tsit5")
- sensealg: Adjoint sensitivity method string (default: "InterpolatingAdjoint")
- integrator_mode: "generic" or "custom_fixed_step" (default: "generic")
- nsteps: Number of steps for custom integrator (default: 4)
- reversible: Use reversible ODE (default: false)
- atol, rtol: ODE solver tolerances (default: 1e-6)
"""
function NeuralODEBlock(d_model::Int, n_heads::Int, d_ff::Int;
                        t0::Float64=0.0, t1::Float64=1.0,
                        solver="Tsit5",
                        sensealg="InterpolatingAdjoint",
                        integrator_mode="generic",
                        nsteps::Int=4,
                        reversible::Bool=false,
                        atol::Float64=1e-6,
                        rtol::Float64=1e-6)
    block = TransformerBlock(d_model, n_heads, d_ff)
    
    # Parse solver
    solver_obj = if solver isa String
        if solver == "Tsit5"
            Tsit5()
        elseif solver == "RK4"
            RK4()
        elseif solver == "Euler"
            Euler()
        else
            Tsit5()  # default
        end
    else
        solver
    end
    
    # Parse sensealg (adjoint sensitivity method)
    # For reversible mode, prefer BacksolveAdjoint for memory efficiency
    sensealg_obj = if reversible && sensealg == "InterpolatingAdjoint"
        # BacksolveAdjoint is more memory-efficient for reversible ODEs
        BacksolveAdjoint(autojacvec=ZygoteVJP(true))
    elseif sensealg == "InterpolatingAdjoint"
        InterpolatingAdjoint(autojacvec=ZygoteVJP(true))
    elseif sensealg == "BacksolveAdjoint"
        BacksolveAdjoint(autojacvec=ZygoteVJP(true))
    elseif sensealg == "QuadratureAdjoint"
        QuadratureAdjoint(autojacvec=ZygoteVJP(true))
    else
        InterpolatingAdjoint(autojacvec=ZygoteVJP(true))  # default
    end
    
    integrator_sym = integrator_mode == "custom_fixed_step" ? :custom_fixed_step : :generic
    
    return NeuralODEBlock(block, (t0, t1), solver_obj, sensealg_obj, 
                         integrator_sym, nsteps, reversible, atol, rtol)
end

"""
    time_embedding(t, d_model) -> Vector{Float32}

Sinusoidal embedding of the scalar integration time `t` into a `d_model`-length
vector. Adding this to the hidden state makes the ODE dynamics genuinely
time-dependent (non-autonomous), so f(h, t) varies along the continuous depth
instead of applying the same map at every t.

Built with broadcasting + `vcat` (no in-place mutation) so it stays
Zygote-compatible, and wrapped at the call site with `@ignore_derivatives`
since it carries no learnable parameters.
"""
function time_embedding(t::Real, d_model::Int)
    half = max(div(d_model, 2), 1)
    # geometric range of frequencies, classic Transformer positional scheme
    freqs = exp.(-(log(10000.0f0)) .* Float32.(0:half-1) ./ Float32(half))
    args = Float32(t) .* freqs
    emb = vcat(sin.(args), cos.(args))
    if length(emb) < d_model
        emb = vcat(emb, zeros(Float32, d_model - length(emb)))
    elseif length(emb) > d_model
        emb = emb[1:d_model]
    end
    return emb
end

# ODE dynamics: treat sequence as part of state; work with flattened vector.
# The time embedding is broadcast across sequence and batch so f depends on t.
function odefunc!(du, u, p, t, block::TransformerBlock, d_model, seq_len, batch)
    # u is a flat vector: length = d_model * seq_len * batch
    x = reshape(u, d_model, seq_len, batch)
    temb = @ignore_derivatives reshape(time_embedding(t, d_model), d_model, 1, 1)
    dx = block(x .+ temb; mask=true)      # same shape
    du .= vec(dx)
end

"""
    continuous_attention_integrator(block, h0, tspan, nsteps)

Custom fixed-step integrator using Runge-Kutta 4th order method.
This provides a tailored integration scheme specifically for Transformer dynamics.

Arguments:
- block: TransformerBlock that defines the dynamics f(h, t)
- h0: Initial hidden state (d_model, seq_len, batch)
- tspan: Time interval (t0, t1)
- nsteps: Number of integration steps

Returns:
- Final hidden state after integration
"""
function continuous_attention_integrator(block::TransformerBlock,
                                         h0::Array{Float32,3},
                                         tspan::Tuple{Float64,Float64},
                                         nsteps::Int)
    t0, t1 = tspan
    dt = Float32((t1 - t0) / nsteps)
    d_model = size(h0, 1)
    h = h0

    # f(h, t) = block(h + time_embedding(t)); each RK4 stage is evaluated at its
    # own time so the integrator sees genuine non-autonomous dynamics.
    f(state, tt) = block(state .+ (@ignore_derivatives reshape(time_embedding(tt, d_model), d_model, 1, 1)); mask=true)

    for i in 1:nsteps
        t = Float32(t0) + (i - 1) * dt

        # RK4 stages evaluated at t, t+dt/2, t+dt/2, t+dt
        k1 = f(h, t)
        k2 = f(h .+ dt/2 .* k1, t + dt/2)
        k3 = f(h .+ dt/2 .* k2, t + dt/2)
        k4 = f(h .+ dt .* k3, t + dt)

        # RK4 update
        h = h .+ dt/6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    end

    return h
end

function (n::NeuralODEBlock)(x)
    # x: (d_model, seq_len, batch)
    d_model, seq_len, batch = size(x)
    
    if n.integrator_mode == :custom_fixed_step
        # Use custom RK4-style integrator
        return continuous_attention_integrator(n.block, x, n.tspan, n.nsteps)
    else
        # Use generic DifferentialEquations solver with adjoint
        u0 = vec(x)
        
        # Create ODE function with closure over block
        dudt!(du, u, p, t) = odefunc!(du, u, p, t, n.block, d_model, seq_len, batch)
        
        prob = ODEProblem(dudt!, u0, n.tspan, nothing)
        
        # Solve with adjoint sensitivity method for efficient backpropagation
        # The sensealg ensures gradients propagate correctly through the ODE integration
        sol = solve(prob, n.solver, 
                   save_everystep=false,
                   sensealg=n.sensealg,
                   abstol=n.atol,
                   reltol=n.rtol)
        
        uT = sol.u[end]
        return reshape(uT, d_model, seq_len, batch)
    end
end

Functors.@functor NeuralODEBlock (block,)

export NeuralODEBlock

end # module NeuralODEBlockModule
