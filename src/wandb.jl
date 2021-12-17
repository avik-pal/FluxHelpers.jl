struct ParameterStateGradientWatcher
    p::Dict
    s::Dict
end

function capture_parameter_state_references(model, base_name=nothing; ps::Zygote.IdSet=Flux.params(model).params)
    parameter_references = Dict{String,AbstractArray}()
    state_references = Dict{String,AbstractArray}()

    fmap_custom(model, base_name) do x, k
        if x isa AbstractArray{<:Number}
            if x ∈ ps
                parameter_references[string(k)] = x
            else
                state_references[string(k)] = x
            end
        end
        return x
    end

    return parameter_references, state_references
end

function ParameterStateGradientWatcher(model, base_name::Symbol=Symbol(); ps::Zygote.IdSet=Flux.params(model).params)
    parameter_references, state_references = capture_parameter_state_references(model, base_name; ps=ps)
    return ParameterStateGradientWatcher(parameter_references, state_references)
end

function Base.log(lg::WandbLogger, ps::ParameterStateGradientWatcher, gs::Zygote.Grads; kwargs...)
    log_dict = Dict{String,Any}()
    for (pname, p) in ps.p
        log_dict["Parameters/" * pname] = Wandb.Histogram(p |> cpu)
        log_dict["Gradients/" * pname] = Wandb.Histogram(gs[p] |> cpu)
    end
    for (sname, s) in ps.s
        log_dict["States/" * sname] = Wandb.Histogram(s |> cpu)
    end
    return log(lg, log_dict; kwargs...)
end
