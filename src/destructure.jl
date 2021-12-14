resolve_keys(key::Symbol, keys_children) = Symbol.((string(key) * "_") .* string.(keys_children))

Flux.Parallel(connection, layers::Vector) = Flux.Parallel(connection, layers...)

function fmap_custom(f, x, k=nothing; exclude=isleaf, cache=IdDict())
    # Because of https://github.com/FluxML/Flux.jl/blob/878b39cfd07b2366cd86f6a2b5b820990968c0ff/src/layers/basic.jl#L44
    k = x isa Chain ? (k === nothing ? Symbol("_layers") : Symbol(string(k) * "_layers")) :
        (k === nothing ? Symbol() : k)

    haskey(cache, x) && return cache[x]
    y = if exclude(x)
        f(x, k)
    else
        func, re = functor(x)
        ks = resolve_keys(k, keys(func))
        res = re(map((x, y) -> first(fmap_custom(f, x, y; exclude=exclude, cache=cache)), values(func), ks))
        res = func isa NTuple ? Tuple(res) : res
    end
    x !== nothing && (cache[x] = (y, k))
    return y, k
end

"""
    destructure_parameters_states(model)
    destructure_parameters_states(model, ps)

# Arguments

Destructures the model and returns 3 values:

    - `parameters`: Trainable parameters of the `model` (by default all arrays in the @functor model are trainable. This can be changed by defining the `Flux.trainable` function for the `model`)
    - `states`: Arrays accessible by `fmap` but not trainable parameters
    - `restructure closure`: Can be used to restore the model given parameter and state vectors

Pass an `IdSet` to `ps` to only include certain `parameters`. Rest will be treated as `states`.

See also: [`destructure_parameters`](@ref)
"""
function destructure_parameters_states(m, ps::Zygote.IdSet=Flux.params(m).params)
    parameters = Zygote.Buffer([])
    states = Zygote.Buffer([])
    parameters_keys = Dict{Symbol,Function}()
    states_keys = Dict{Symbol,Function}()

    fmap_custom(m) do x, k
        if x isa AbstractArray{<:Number}
            v = vec(x)
            if x ∈ ps
                parameters_keys[k] = () -> zero(v)
                push!(parameters, v)
            else
                states_keys[k] = () -> zero(v)
                push!(states, v)
            end
        end
        return x
    end

    return (vcat(copy(parameters)...), vcat(copy(states)...),
            (p, s) -> _restructure_parameters_states(m, parameters_keys, states_keys, p, s))
end

function destructure_parameters_states(m, parameters_keys, states_keys)
    parameters = Zygote.Buffer([])
    states = Zygote.Buffer([])

    fmap_custom(m) do x, k
        if x === nothing
            if k ∈ keys(parameters_keys)
                push!(parameters, parameters_keys[k]())
            elseif k ∈ keys(states_keys)
                push!(states, states_keys[k]())
            end
        elseif x isa AbstractArray{<:Number}
            if k ∈ keys(parameters_keys)
                push!(parameters, vec(x))
            elseif k ∈ keys(states_keys)
                push!(states, vec(x))
            else
                error("Unknown Key Encountered: " * k)
            end
        end
        return x
    end

    return (vcat(copy(parameters)...), vcat(copy(states)...),
            (p, s) -> _restructure_parameters_states(m, parameters_keys, states_keys, p, s))
end

function _restructure_parameters_states(m, parameters_keys, states_keys, parameters, states)
    i, j = 0, 0

    m̄ = fmap_custom(m) do x, k
        !(x isa AbstractArray{<:Number}) && return x
        if k ∈ keys(parameters_keys)
            x = reshape(parameters[i .+ (1:length(x))], size(x))
            i += length(x)
        elseif k ∈ keys(states_keys)
            x = reshape(states[j .+ (1:length(x))], size(x))
            j += length(x)
        else
            error("Unknown Key Encountered: " * k)
        end
        return x
    end[1]

    exp_vector_size = i + j
    recv_vector_size = length(parameters) + length(states)
    recv_vector_size == exp_vector_size || @warn "Expected $(exp_vector_size) params, got $(recv_vector_size)"
    return m̄
end

Zygote.@adjoint function _restructure_parameters_states(m, parameters_keys, states_keys, parameters, states)
    m̄ = _restructure_parameters_states(m, parameters_keys, states_keys, parameters, states)
    numel = length(parameters) + length(states)
    function _restructure_parameters_states_pullback(dm)
        ∇parameters, ∇states, _ = destructure_parameters_states(dm, parameters_keys, states_keys)
        recv_numel = length(∇parameters) + length(∇states)
        numel != recv_numel && @warn "Expected $(numel) params, got $(recv_numel)"
        return (nothing, nothing, nothing, ∇parameters, ∇states)
    end
    return m̄, _restructure_parameters_states_pullback
end

"""
    destructure_parameters(model)
    destructure_parameters(model, ps)

# Arguments

Destructures the model and returns 2 values:

    - `parameters`: Trainable parameters of the `model` (by default all arrays in the @functor model are trainable. This can be changed by defining the `Flux.trainable` function for the `model`)
    - `restructure closure`: Can be used to restore the model given parameter and state vectors

Pass an `IdSet` to `ps` to only include certain `parameters`.

See also: [`destructure_parameters_states`](@ref)
"""
function destructure_parameters(m, ps::Zygote.IdSet=Flux.params(m).params)
    parameters = Zygote.Buffer([])
    parameters_keys = Dict{Symbol,Function}()

    fmap_custom(m) do x, k
        if x isa AbstractArray{<:Number}
            if x ∈ ps
                v = vec(x)
                parameters_keys[k] = () -> zero(v)
                push!(parameters, v)
            end
        end
        return x
    end

    return (vcat(copy(parameters)...), p -> _restructure_parameters(m, parameters_keys, p))
end

function destructure_parameters(m, parameters_keys)
    parameters = Zygote.Buffer([])

    fmap_custom(m) do x, k
        if x === nothing
            k ∈ keys(parameters_keys) && push!(parameters, parameters_keys[k]())
        elseif x isa AbstractArray{<:Number}
            k ∈ keys(parameters_keys) && push!(parameters, vec(x))
        end
        return x
    end

    return (vcat(copy(parameters)...), p -> _restructure_parameters(m, parameters_keys, p))
end

function _restructure_parameters(m, parameters_keys, parameters)
    i = 0

    m̄ = fmap_custom(m) do x, k
        !(x isa AbstractArray{<:Number}) && return x
        if k ∈ keys(parameters_keys)
            x = reshape(parameters[i .+ (1:length(x))], size(x))
            i += length(x)
        end
        return x
    end[1]

    exp_vector_size = i
    recv_vector_size = length(parameters)
    recv_vector_size == exp_vector_size || @warn "Expected $(exp_vector_size) params, got $(recv_vector_size)"
    return m̄
end

Zygote.@adjoint function _restructure_parameters(m, parameters_keys, parameters)
    m̄ = _restructure_parameters(m, parameters_keys, parameters)
    numel = length(parameters)
    function _restructure_parameters_pullback(dm)
        ∇parameters, _ = destructure_parameters(dm, parameters_keys)
        recv_numel = length(∇parameters)
        numel != recv_numel && @warn "Expected $(numel) params, got $(recv_numel)"
        return (nothing, nothing, ∇parameters)
    end
    return m̄, _restructure_parameters_pullback
end
