struct BatchNorm{F1,F2,F3,N} <: ExplicitLayer
    λ::F1
    ϵ::N
    momentum::N
    chs::Int
    initβ::F2
    initγ::F3
    affine::Bool
    track_stats::Bool
end

function BatchNorm(
    chs::Int, λ=identity; initβ=zeros32, initγ=ones32, affine=true, track_stats=true, ϵ=1.0f-5, momentum=0.1f0
)
    return BatchNorm(λ, ϵ, momentum, chs, initβ, initγ, affine, track_stats)
end

initialparameters(l::BatchNorm) = l.affine ? (γ = l.initγ(l.chs), β = l.initβ(l.chs)) : NamedTuple()
initialstates(l::BatchNorm) = (μ=zeros32(l.chs), σ²=ones32(l.chs), training=true)

function Base.show(io::IO, l::BatchNorm)
    print(io, "BatchNorm($(l.chs)")
    (l.λ == identity) || print(io, ", $(l.λ)")
    l.affine || print(io, ", affine=false")
    return print(io, ")")
end

function batchnorm_fallback(
    BN::BatchNorm, x::AbstractArray{T,N}, ps::NamedTuple, states::NamedTuple
) where {T,N}
    @assert size(x, ndims(x) - 1) == BN.chs
    reduce_dims = [1:(N - 2); N]
    affine_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
    return norm_forward(BN, ps, states, x, reduce_dims, affine_shape)
end

function (BN::BatchNorm)(x::AbstractArray{T}, ps::NamedTuple, states::NamedTuple) where {T}
    return batchnorm_fallback(BN, x, ps, states)
end
