function save_flux_model(filepath::String, p::AbstractVector, s::AbstractVector, re)
    base_name, ext = splitext(filepath)
    if ext == ".jld2"
        save(filepath, Dict("model" => re(p, s) |> cpu))
        @info "Saved model to " * filepath
        return nothing
    end
    dict = Dict("parameters" => p |> cpu, "states" => s |> cpu)
    ps_filepath = base_name * ".ps" * ext
    serialize(ps_filepath, dict)
    @info "Model Parameters and States saved to " * ps_filepath
    re_filepath = base_name * ".re" * ext
    serialize(re_filepath, re)
    @info "Restructure Function saved to " * re_filepath
    return nothing
end

save_flux_model(filepath::String, m) = save_flux_model(filepath, destructure_parameters_states(m)...)


function load_flux_model(filepath::String, re=nothing)
    base_name, ext = splitext(filepath)
    if ext == ".jld2"
        model = load(filepath, "model")
        @info "Model loaded from " * filepath
        re === nothing && return model
        p, s, _ = destructure_parameters_states(model)
        return re(p, s)
    end
    ps_filepath = base_name * ".ps" * ext
    re_filepath = base_name * ".re" * ext
    ps = deserialize(ps_filepath)
    parameters, states = ps["parameters"], ps["states"]
    if re === nothing
        try
            re = deserialize(re_filepath)
        catch e
            @warn "Unable to load the restructure function from " * re_filepath * ". Please pass it manually."
            rethrow(e)
        end
    end
    return re(parameters, states)
end


# mutable struct ModelCheckpointer
#     checkpoint_frequency::Int
#     step::Int
#     checkpoint_filepath::String
#     maximum_checkpoints::Int
#     checkpoint_counter::Int

# end