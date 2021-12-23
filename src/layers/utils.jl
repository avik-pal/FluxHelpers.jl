# Hack around Flux to allow multiple inputs to `Chain`
Flux.applychain(::Tuple{}, x1, xs...) = (x1, xs...)
Flux.applychain(::Tuple{}, x::Tuple) = x
Flux.applychain(fs::Tuple, x1, xs...) = Flux.applychain(Base.tail(fs), first(fs)(x1, xs...))
Flux.applychain(fs::Tuple, x::Tuple) = Flux.applychain(Base.tail(fs), first(fs)(x...))

(c::Chain)(x1, xs...) = Flux.applychain(Tuple(c.layers), x1, xs...)
(c::Chain)(x::Tuple) = c(x...)

# Some wrappers around common functions for prettier printing
"""
    ReshapeLayer(dims)

Reshapes the passed arrays to have a size of `(dims..., :)`
"""
struct ReshapeLayer{D}
    dims::D
end

(r::ReshapeLayer)(x) = reshape(x, r.dims..., :)
(r::ReshapeLayer)(xs...) = r.(xs)

"""
    FlattenLayer()

Flattens the passed arrays to 2D matrices.
"""
struct FlattenLayer end

(f::FlattenLayer)(x::AbstractArray{T,N}) where {T,N} = reshape(x, :, size(x, N))
(f::FlattenLayer)(xs...) = f.(xs)

"""
    SelectDim(dim, i)

See the documentation for `selectdim` for more information.
"""
struct SelectDim{I}
    dim::Int
    i::I
end

(s::SelectDim)(x) = selectdim(x, s.dim, s.i)
(s::SelectDim)(xs...) = s.(xs)

"""
    NoOpLayer()

As the name suggests. Does nothing. Better to use this than
`identity` since it takes in multiple inputs
"""
struct NoOpLayer end

(noop::NoOpLayer)(x...) = x
(noop::NoOpLayer)(x) = x
