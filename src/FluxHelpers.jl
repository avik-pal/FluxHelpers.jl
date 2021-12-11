module FluxHelpers

using Flux
using Functors
using Zygote

using Functors: isleaf, children, _default_walk, functor

include("destructure.jl")

export destructure_parameters_states, destructure_parameters

end
