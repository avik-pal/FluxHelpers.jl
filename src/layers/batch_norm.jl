struct BatchNormV2{F,V,N,W}
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

Flux.hasaffine(gn::BatchNormV2) = Flux.hasaffine(gn.attrs)
Flux._isactive(gn::BatchNormV2) = Flux._isactive(gn.attrs)

Flux.@functor BatchNormV2

Flux.trainable(gn::BatchNormV2) = hasaffine(gn) ? (gn.β, gn.γ) : ()

function BatchNormV2(chs::Int, λ=identity; initβ=zeros32, initγ=ones32, affine=true, track_stats=false, ϵ=1.0f-5,
                     momentum=0.1f0)
    # NOTE: We need to initialize these since CUDNN requires them
    β = initβ(chs)
    γ = initγ(chs)
    μ = zeros32(chs)
    σ² = ones32(chs)

    return BatchNormV2(λ, β, γ, μ, σ², ϵ, momentum, chs, NormAttributes(affine, track_stats, nothing))
end

function (BN::BatchNormV2)(x::AbstractArray{T,N}) where {T,N}
    # @assert size(x, ndims(x)-1) == BN.chs
    reduce_dims = [1:(N - 2); N]
    affine_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
    return norm_forward(BN, x, reduce_dims, affine_shape)
end

function testmode!(m::BatchNormV2, mode=true)
    return (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)
end

function Base.show(io::IO, l::BatchNormV2)
    print(io, "BatchNormV2($(l.chs)")
    (l.λ == identity) || print(io, ", $(l.λ)")
    hasaffine(l) || print(io, ", affine=false")
    return print(io, ")")
end

function (BN::BatchNormV2)(x::Union{CuArray{T,2},CuArray{T,4},CuArray{T,5}},
                           cache=nothing) where {T<:Union{Float32,Float64}}
    return BN.λ.(batchnorm(BN.γ, BN.β, x, BN.μ, BN.σ², BN.momentum; cache=cache, alpha=1, beta=0, eps=BN.ϵ,
                           training=Flux._isactive(BN)))
end
