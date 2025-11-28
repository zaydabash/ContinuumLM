"""
    Models.jl

Top-level models module that includes all model components.
"""
module Models

include("Embeddings.jl")
include("Attention.jl")
include("ContinuousTransformer.jl")
include("NeuralODEBlock.jl")
include("LanguageModel.jl")

using .Embeddings
using .Attention
using .ContinuousTransformer
using .NeuralODEBlock
using .LanguageModel

export build_model, LanguageModel, LanguageModelStruct, LMState, init_state, generate_step

"""
    build_model(mc::Config.ModelConfig)

Construct a language model according to the configuration.
"""
function build_model(mc)
    return LanguageModel.build_language_model(mc)
end

end # module

