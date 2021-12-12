module FluxExperimental

using CUDA
using FileIO
using Flux
using Functors
using Serialization
using Zygote

using Functors: isleaf, children, _default_walk, functor

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

export destructure_parameters_states, destructure_parameters
export enable_fast_mode!
export save_flux_model, load_flux_model

end
