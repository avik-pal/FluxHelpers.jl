"""
    GroupNormV2(channels::Integer, G::Integer, λ=identity;
                initβ=zeros32, initγ=ones32,
                affine=true, track_stats=false,
                ϵ=1f-5, momentum=0.1f0)

Same as [`Flux.GroupNorm`](@ref) but works well with `destructure_parameters` & `destructure_parameters_states`
"""
struct GroupNormV2{F,V,N,W}
    G::Int  # number of groups
    λ::F  # activation function
    β::V  # bias
    γ::V  # scale
    μ::W     # moving mean
    σ²::W    # moving std
    ϵ::N
    momentum::N
    chs::Int # number of channels
    attrs::NormAttributes
end

Flux.hasaffine(gn::GroupNormV2) = Flux.hasaffine(gn.attrs)
Flux._isactive(gn::GroupNormV2) = Flux._isactive(gn.attrs)

Flux.@functor GroupNormV2

Flux.trainable(gn::GroupNormV2) = hasaffine(gn) ? (gn.β, gn.γ) : ()

function GroupNormV2(chs::Int, G::Int, λ=identity; initβ=zeros32, initγ=ones32, affine=true, track_stats=false, ϵ=1f-5,
                     momentum=0.1f0)
    @assert chs % G == 0 "The number of groups ($(G)) must divide the number of channels ($chs)"

    β = affine ? initβ(chs) : nothing
    γ = affine ? initγ(chs) : nothing
    μ = track_stats ? zeros32(G) : nothing
    σ² = track_stats ? ones32(G) : nothing

    return GroupNormV2(G, λ, β, γ, μ, σ², ϵ, momentum, chs, NormAttributes(affine, track_stats, nothing))
end

function (gn::GroupNormV2)(x::AbstractArray{T,N}) where {T,N}
    # Not doing assertion checks
    # @assert N > 2
    # @assert size(x, N - 1) == gn.chs
    sz = size(x)
    x_2 = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ gn.G, gn.G, sz[N])
    N_ = ndims(x_2)
    reduce_dims = 1:(N_ - 2)
    affine_shape = ntuple(i -> i ∈ (N_ - 1, N_ - 2) ? size(x_2, i) : 1, N_)
    return reshape(norm_forward(gn, x_2, reduce_dims, affine_shape), sz)
end

function testmode!(m::GroupNormV2, mode=true)
    return (m.attrs.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)
end

function Base.show(io::IO, l::GroupNormV2)
    print(io, "GroupNormV2($(l.chs), $(l.G)")
    l.λ == identity || print(io, ", ", l.λ)
    hasaffine(l) || print(io, ", affine=false")
    return print(io, ")")
end
