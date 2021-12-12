"""
    WeightNorm(layer; dim::Union{Tuple,Vector,Int,Nothing}=nothing, ps=Flux.params(layer))

Applies weight normalization to `ps` parameters in the given layer:

``w = g\\frac{v}{||v||}``

!!!note:
    This assumes that `layer` cannot contain non-trainable states.

Ref: [`https://arxiv.org/abs/1602.07868`](@ref)
"""
struct WeightNorm{Re,P,S,D}
    layer_re::Re
    parameters::P
    states::S
    dims::D
end

Flux.@functor WeightNorm (parameters, states)

function Base.show(io::IO, wn::WeightNorm)
    ps = Flux.params(wn)
    p = update_parameters(wn)
    l = wn.layer_re(p, wn.states)
    print(io, "WeightNorm(")
    print(io, l)
    return print(") ", string(sum(length.(ps))), " Trainable Parameters")
end

function WeightNorm(layer; dim::Union{Tuple,Vector,Int,Nothing}=nothing, ps=Flux.params(layer))
    if ps isa Array
        ps = Flux.params(ps)
    end
    dim = dim === nothing ? [ndims(p) for p in ps] : (dim isa Int ? [dim for _ in 1:length(ps)] : dim)

    p_, s_, layer_re = destructure_parameters_states(layer, ps.params)

    parameters = []
    for (i, p) in enumerate(ps)
        g_val = _norm(p, dim[i])
        v_val = copy(p)
        push!(parameters, (g_val, v_val))
    end

    return WeightNorm(layer_re, tuple(parameters...), s_, dim)
end

@inline compute_normed_weight(v, g, dim) = v .* (g ./ _norm(v, dim))

@inline function update_parameters(wn::WeightNorm)
    return vcat(ntuple(i -> vec(compute_normed_weight(wn.parameters[i][2], wn.parameters[i][1], wn.dims[i])),
                       length(wn.dims))...)
end

(wn::WeightNorm)(args...; kwargs...) = wn.layer_re(update_parameters(wn),wn.states)(args...; kwargs...)

@inline _norm(x; dims=Colon()) = sqrt.(sum(abs2, x; dims=dims))

# Compute norm over all dimensions except `except_dim`
@inline _norm(x::AbstractArray{T,N}, except_dim) where {T,N} = _norm(x; dims=filter(i -> i != except_dim, 1:N))
