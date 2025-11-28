"""
    runtests.jl

Test suite for Neural ODE Language Model.
"""

using Pkg
Pkg.activate(@__DIR__ * "/..")

using Test

include("test_data.jl")
include("test_models.jl")
include("test_training.jl")

@testset "NeuralODELM Tests" begin
    @testset "Data" begin
        test_data()
    end
    
    @testset "Models" begin
        test_models()
    end
    
    @testset "Training" begin
        test_training()
    end
end

