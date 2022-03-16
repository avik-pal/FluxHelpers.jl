struct Chain{T} <: ExplicitLayer
    layers::T
    function Chain(xs...)
        xs = flatten_model(xs)
        return new{typeof(xs)}(xs)
    end
    Chain(xs::AbstractVector) = Chain(xs...)
end

function flatten_model(layers::Union{AbstractVector,Tuple})
    new_layers = []
    for l in layers
        f = flatten_model(l)
        if f isa Tuple || f isa AbstractVector
            append!(new_layers, f)
        elseif f isa Chain
            append!(new_layers, f.layers)
        else
            push!(new_layers, f)
        end
    end
    return layers isa AbstractVector ? new_layers : Tuple(new_layers)
end

flatten_model(x) = x

function initialparameters(c::Chain)
    return (; zip(ntuple(i -> Symbol("layer_$i"), length(c.layers)), initialparameters.(c.layers))...)
end

function initialstates(c::Chain)
    return (; zip(ntuple(i -> Symbol("layer_$i"), length(c.layers)), initialstates.(c.layers))...)
end

(c::Chain)(x, ps::NamedTuple, s::NamedTuple) = applychain(c.layers, x, ps, s)

@generated function applychain(layers::Tuple{Vararg{<:Any,N}}, x, ps, s) where {N}
    symbols = vcat(:x, [gensym() for _ in 1:N])
    calls = [:($(symbols[i+1]) = layers[$i]($(symbols[i]), ps[$i], s[$i])) for i in 1:N]
    return Expr(:block, calls...)
end
