"""
    SpectralNorm(layer; ps=Flux.params(layer), power_iterations::Int=1)

Applies spectral normalization to `ps` parameters in the given layer.

!!!note:
This assumes that `layer` cannot contain non-trainable states.

Ref: [`https://arxiv.org/abs/1802.05957`](@ref)
"""
struct SpectralNorm{Re,P,S,PUV}
    layer_re::Re
    parameters::P
    states::S
    parameter_uvs::PUV
    power_iterations::Int
end

Flux.@functor SpectralNorm (parameters, states)

function Base.show(io::IO, sn::SpectralNorm)
    ps = Flux.params(sn)
    p = update_parameters(sn)
    l = wn.layer_re(p, sn.states)
    print(io, "SpectralNorm(")
    print(io, l)
    return print(") ", string(sum(length.(ps))), " Trainable Parameters")
end

l2normalize(x::AbstractArray{T}) where {T} = x ./ (norm(x, 2) + eps(T))

function construct_uv(x::AbstractArray)
    H, W = Flux.flatten(x)
    u = l2normalize(randn(eltype(x), H))
    v = l2normalize(randn(eltype(x), W))
    return (u, v)
end

function construct_uv(x::CuArray)
    H, W = Flux.flatten(x)
    u = l2normalize(CUDA.randn(eltype(x), H))
    v = l2normalize(CUDA.randn(eltype(x), W))
    return (u, v)
end

function spectral_normalization(x, u, v, power_iterations::Int=1)
    x_mat = Flux.flatten(x)
    for _ in 1:power_iterations
        v .= l2normalize(x_mat' * u)
        u .= l2normalize(x_mat * v)
    end
    Σ = sum(u .* (x_mat * v))
    return Σ
end

Zygote.@nograd spectral_normalization

function update_parameters(sn::SpectralNorm)
    return vcat(ntuple(i -> vec(sn.parameters[i] ./
                                spectral_normalization(sn.parameters[i], sn.parameter_uvs[i][1], sn.parameter_uvs[i][2],
                                                       sn.power_iterations)), length(sn.dims))...)
end

function SpectralNorm(layer; ps=Flux.params(layer), power_iterations::Int=1)
    if ps isa Array
        ps = Flux.params(ps)
    end

    _, s_, layer_re = destructure_parameters_states(layer, ps.params)

    parameters = []
    parameter_uvs = []
    foreach(p -> begin
                push!(parameter_uvs, construct_uv(p))
                push!(parameter, p)
            end, ps)

    return SpectralNorm(layer_re, tuple(parameters...), s_, tuple(parameter_uvs...), power_iterations)
end

(sn::SpectralNorm)(args...; kwargs...) = sn.layer_re(update_parameters(sn), sn.states)(args...; kwargs...)
