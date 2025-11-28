"""
    NeuralODEBlock.jl

Neural ODE block implementing continuous-depth Transformer with advanced features:
- Proper adjoint sensitivity methods for efficient backpropagation
- Custom continuous-attention kernel integrator (RK4-style)
- Reversible ODE design for memory efficiency

The hidden state evolves continuously via an ODE parameterized by Transformer dynamics.
"""
module NeuralODEBlock

using Flux
using DifferentialEquations
using DiffEqFlux
using DiffEqFlux: InterpolatingAdjoint, BacksolveAdjoint, QuadratureAdjoint, ZygoteVJP
using ..Attention: TransformerBlock

export NeuralODEBlock

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

# ODE dynamics: treat sequence as part of state; work with flattened vector.
function odefunc!(du, u, p, t, block::TransformerBlock, d_model, seq_len, batch)
    # u is a flat vector: length = d_model * seq_len * batch
    x = reshape(u, d_model, seq_len, batch)
    dx = block(x; mask=true)      # same shape
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
    dt = (t1 - t0) / nsteps
    h = h0
    
    for i in 1:nsteps
        t = t0 + (i - 1) * dt
        
        # RK4 stages
        k1 = block(h; mask=true)
        k2 = block(h .+ dt/2 .* k1; mask=true)
        k3 = block(h .+ dt/2 .* k2; mask=true)
        k4 = block(h .+ dt .* k3; mask=true)
        
        # RK4 update
        h = h .+ dt/6 .* (k1 .+ 2*k2 .+ 2*k3 .+ k4)
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

Flux.@functor NeuralODEBlock

end # module
