module FluxExperimental

using CUDA
using FileIO
using Flux
using Functors
using LinearAlgebra
using Requires
using Serialization
using Statistics
using Wandb
using Zygote

using Flux: hasaffine, ones32, zeros32, _isactive
using Functors: isleaf, children, _default_walk, functor
import NNlibCUDA: batchnorm, ∇batchnorm

function __init__()
    @require ChemistryFeaturization = "6c925690-434a-421d-aea7-51398c5b007a" begin
        include("graph_data.jl")
        # Atomic Graph Net Layers
        export batch_graph_data, BatchedAtomicGraph
    end
end

on_gpu(arr::CuArray) = true
on_gpu(arr::AbstractArray) = false

"""
    on_gpu(layer)

Checks if the parameters and states of the layer is in GPU memory. If one is in GPU and other is not,
then a warning is displayed. If there are no parameters and states then false is returned by default.
"""
on_gpu(layer) = on_gpu(destructure_parameters_states(layer)[1:2]...)

function on_gpu(p, s)
    if length(p) > 0 && length(s) > 0
        p_device = on_gpu(p)
        s_device = on_gpu(s)
        if !(p_device == s_device)
            @warn "Layer parameters and states are not on the same device"
        end
        return p_device && s_device
    elseif length(p) == 0
        return on_gpu(s)
    elseif length(s) == 0
        return on_gpu(p)
    else
        return false
    end
end

abstract type AbstractFluxLayer end

function Base.show(io::IO, l::AbstractFluxLayer)
    p, s, _ = destructure_parameters_states(l)
    device = on_gpu(p, s) ? "GPU" : "CPU"
    return print(io, string(typeof(l).name.name), "() ", string(length(p)), " Trainable Parameters & ",
                 string(length(s)), " States & Device = ", device)
end

"""
    debug_backward_pass(msg::String)

Prints the `msg` during the forward pass. During the backwards pass prints `∇(msg)`. Also prints
the time it took to reach the backward pass since the forward pass happended.
"""
@inline function debug_backward_pass(msg::String)
    start_time = Zygote.@ignore time()
    return Zygote.hook(Δ -> begin
                           println("∇(" * msg * ") -- Took $(time() - start_time) s")
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

# Needed for destructure
include("layers/fchain.jl")

include("destructure.jl")
include("saving.jl")
include("utils.jl")

# Useful Layer Implementations
include("layers/norm_utils.jl")
include("layers/functional_wrappers.jl")
include("layers/weight_norm.jl")
include("layers/spectral_norm.jl")
include("layers/dropout.jl")
include("layers/utils.jl")
include("layers/agn.jl")
include("layers/group_norm.jl")
include("layers/batch_norm.jl")

# Extending the logging capabilities of wandb
include("wandb.jl")

# Common Utilities
export destructure_parameters_states, destructure_parameters
export enable_fast_mode!
export save_flux_model, load_flux_model
export debug_backward_pass
# Layers
export conv1x1, conv3x3, conv5x5, conv1x1_norm, conv3x3_norm, conv5x5_norm, conv_norm, downsample_module,
       upsample_module
export FChain
export WeightNorm, SpectralNorm
export VariationalHiddenDropout, update_is_variational_hidden_dropout_mask_reset_allowed
export AGNConv, AGNMaxPool, AGNMeanPool
export GroupNormV2, BatchNormV2
export ReshapeLayer, FlattenLayer, SelectDim, NoOpLayer
# Logging
export ParameterStateGradientWatcher


end
