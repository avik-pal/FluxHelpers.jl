module ExplicitLayers

using ZygoteRules, FastBroadcast, Statistics, Zygote, NNlib
import Flux: zeros32, ones32, glorot_normal, glorot_uniform

# Base Type
abstract type ExplicitLayer end

initialparameters(::ExplicitLayer) = NamedTuple()
initialstates(::ExplicitLayer) = NamedTuple()

init(l::ExplicitLayer) = (initialparameters(l), initialstates(l))

nestedtupleofarrayslength(t::Any) = 1
nestedtupleofarrayslength(t::AbstractArray) = length(t)
function nestedtupleofarrayslength(t::Union{NamedTuple,Tuple})
    length(t) == 0 && return 0
    return sum(nestedtupleofarrayslength, t)
end

parameterlength(l::ExplicitLayer) = nestedtupleofarrayslength(initialparameters(l))
statelength(l::ExplicitLayer) = nestedtupleofarrayslength(initialstates(l))

# Utilities
include("norm_utils.jl")

# Layer Implementations
include("chain.jl")
include("batchnorm.jl")
include("linear.jl")

end