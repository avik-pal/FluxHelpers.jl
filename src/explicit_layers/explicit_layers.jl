module ExplicitLayers

using ZygoteRules, FastBroadcast, Statistics, Zygote, NNlib, CUDA, Random, Setfield
import Flux: zeros32, ones32, glorot_normal, glorot_uniform, convfilter, expand, calc_padding, DenseConvDims, _maybetuple_string, reshape_cell_output

# Base Type
abstract type ExplicitLayer end

initialparameters(::AbstractRNG, ::ExplicitLayer) = NamedTuple()
initialparameters(l::ExplicitLayer) = initialparameters(Random.GLOBAL_RNG, l)
initialstates(::AbstractRNG, ::ExplicitLayer) = NamedTuple()
initialstates(l::ExplicitLayer) = initialstates(Random.GLOBAL_RNG, l)

init(rng::AbstractRNG, l::ExplicitLayer) = (initialparameters(rng, l), initialstates(rng, l))
init(l::ExplicitLayer) = init(Random.GLOBAL_RNG, l)

nestedtupleofarrayslength(t::Any) = 1
nestedtupleofarrayslength(t::AbstractArray) = length(t)
function nestedtupleofarrayslength(t::Union{NamedTuple,Tuple})
    length(t) == 0 && return 0
    return sum(nestedtupleofarrayslength, t)
end

parameterlength(l::ExplicitLayer) = nestedtupleofarrayslength(initialparameters(l))
statelength(l::ExplicitLayer) = nestedtupleofarrayslength(initialstates(l))

# Test Mode
function testmode(states::NamedTuple, mode::Bool = true)
    updated_states = []
    for (k, v) in pairs(states)
        if k == :training
            push!(updated_states, k => !mode)
            continue
        end
        push!(updated_states, k => testmode(v, mode))
    end
    return (; updated_states...)
end

testmode(x::Any, mode::Bool = true) = x

testmode(m::ExplicitLayer, mode::Bool = true) = testmode(initialstates(m), mode)

trainmode(x::Any, mode::Bool = true) = testmode(x, !mode)

# Utilities
zeros32(rng::AbstractRNG, args...; kwargs...) = zeros32(args...; kwargs...)
ones32(rng::AbstractRNG, args...; kwargs...) = ones32(args...; kwargs...)

include("norm_utils.jl")

# Layer Implementations
include("chain.jl")
include("batchnorm.jl")
include("linear.jl")
include("convolution.jl")
include("weightnorm.jl")
include("wrappers.jl")


end