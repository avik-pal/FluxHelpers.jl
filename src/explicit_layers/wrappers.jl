struct ReshapeLayer{N} <: ExplicitLayer
    dims::NTuple{N,Int}
end

(r::ReshapeLayer)(x::AbstractArray, ::NamedTuple, ::NamedTuple) = reshape(x, r.dims..., :)

struct FlattenLayer <:ExplicitLayer end

(f::FlattenLayer)(x::AbstractArray{T,N}, ::NamedTuple, ::NamedTuple) where {T,N} = reshape(x, :, size(x, N))

struct SelectDim{I} <: ExplicitLayer
    dim::Int
    i::I
end

(s::SelectDim)(x, ::NamedTuple, ::NamedTuple) = selectdim(x, s.dim, s.i)

struct NoOpLayer <: ExplicitLayer end

(noop::NoOpLayer)(x, ::NamedTuple, ::NamedTuple) = x