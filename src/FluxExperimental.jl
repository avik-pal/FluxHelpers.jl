module FluxExperimental

using CUDA
using FileIO
using Flux
using Functors
using Requires
using Serialization
using Statistics
using Zygote

using Functors: isleaf, children, _default_walk, functor

function __init__()
    @require ChemistryFeaturization="6c925690-434a-421d-aea7-51398c5b007a" begin
        include("graph_data.jl")
        # Atomic Graph Net Layers
        export batch_graph_data, BatchedAtomicGraph
    end
end

"""
    debug_backward_pass(msg::String)

Prints the `msg` during the forward pass. During the backwards pass prints `∇(msg)`.
"""
@inline function debug_backward_pass(msg::String)
    return Zygote.hook(Δ -> begin
                           println("∇(" * msg * ")")
                           return Δ
                       end, println(msg))
end


"""
    enable_fast_mode!()

Enable CUDA.FAST_MATH mode. Note this will reduce floating point precision but we rarely care about that.
"""
function enable_fast_mode!()
    CUDA.math_mode!(CUDA.FAST_MATH)
    return nothing
end

include("destructure.jl")
include("saving.jl")

include("layers/functional_wrappers.jl")
include("layers/weight_norm.jl")
include("layers/spectral_norm.jl")
include("layers/dropout.jl")
include("layers/utils.jl")
include("layers/agn.jl")
include("layers/normalize.jl")


# Common Utilities
export destructure_parameters_states, destructure_parameters
export enable_fast_mode!
export save_flux_model, load_flux_model
export debug_backward_pass
# Layers
export conv1x1, conv3x3, conv5x5, conv1x1_norm, conv3x3_norm, conv5x5_norm, conv_norm,
       downsample_module, upsample_module
export WeightNorm, SpectralNorm
export VariationalHiddenDropout, update_is_variational_hidden_dropout_mask_reset_allowed
export AGNConv, AGNMaxPool, AGNMeanPool
export GroupNormV2

end
