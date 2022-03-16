struct Dense{bias,F1,F2,F3} <: ExplicitLayer
    λ::F1
    in_dims::Int
    out_dims::Int
    initW::F2
    initb::F3
end

function Dense(mapping::Pair{Int,Int}, λ=identity; initW=glorot_uniform, initb=zeros32, bias::Bool=true)
    return Dense{bias,typeof(λ),typeof(initW),typeof(initb)}(λ, first(mapping), last(mapping), initW, initb)
end

initialparameters(d::Dense{true}) = (weight = d.initW(d.out_dims, d.in_dims), bias = d.initb(d.out_dims, 1))
initialparameters(d::Dense{false}) = (weight = d.initW(d.out_dims, d.in_dims),)

(d::Dense{false})(x::AbstractVecOrMat{T}, ps::NamedTuple, ::NamedTuple) where {T} = (NNlib.fast_act(d.λ, x)).(ps.weight * x)

(d::Dense{true})(x::AbstractMatrix{T}, ps::NamedTuple, ::NamedTuple) where {T} = (NNlib.fast_act(d.λ, x)).(ps.weight * x .+ ps.bias)

(d::Dense{true})(x::AbstractVector{T}, ps::NamedTuple, ::NamedTuple) where {T} = (NNlib.fast_act(d.λ, x)).(ps.weight * x .+ vec(ps.bias))
