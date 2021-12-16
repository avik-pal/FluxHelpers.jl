mutable struct NormAttributes
    affine::Bool
    track_stats::Bool
    active::Union{Bool,Nothing}
end

Flux.hasaffine(attr::NormAttributes) = attr.affine

function get_stats(::Val{true}, ::Val{false}, l, x::AbstractArray{T,N}, reduce_dims) where {T,N}
    # testmode with tracked stats
    stats_shape = ntuple(i -> i == N - 1 ? size(x, N - 1) : 1, N)
    return reshape(l.μ, stats_shape), reshape(l.σ², stats_shape)
end

function get_stats(::Val{false}, active, l, x, reduce_dims)
    # trainmode or testmode without tracked stats
    μ = mean(x; dims=reduce_dims)
    diff = x .- μ
    return μ, mean(abs2, diff; dims=reduce_dims)
end

function get_stats(::Val{true}, active::Val{true}, l, x::AbstractArray{T,N}, reduce_dims) where {T,N}
    # trainmode with tracked stats
    μ, σ² = get_stats(Val(false), active, l, x, reduce_dims)
    Zygote.ignore() do
        # FIXME: Sync for FluxMPI
        mtm = l.momentum
        m = prod(size(x)[reduce_dims])  # needed for computing corrected var
        μnew = vec(N ∈ reduce_dims ? μ : mean(μ; dims=N))
        σ²new = vec(N ∈ reduce_dims ? σ² : mean(σ²; dims=N))
        l.μ .= (1 - mtm) .* l.μ .+ mtm .* μnew
        return l.σ² .= (1 - mtm) .* l.σ² .+ mtm .* (m / (m - one(eltype(l.σ²)))) .* σ²new
    end
    return μ, σ²
end

function norm_forward(l, x::AbstractArray{T,N}, reduce_dims, affine_shape) where {T,N}
    μ, σ² = get_stats(Val(l.attrs.track_stats), Val(_isactive(l)), l, x, reduce_dims)
    if hasaffine(l)
        γ = reshape(l.γ, affine_shape)
        β = reshape(l.β, affine_shape)
        return l.λ.(_norm_forward(μ, σ², x, γ, β, l.ϵ))
    else
        return l.λ.(_norm_forward(μ, σ², x, l.ϵ))
    end
end

_norm_forward(μ, σ², x, ϵ) = (x .- μ) ./ sqrt.(σ² .+ ϵ)

_norm_forward(μ, σ², x, γ, β, ϵ) = γ .* (x .- μ) ./ sqrt.(σ² .+ ϵ) .+ β

Zygote.@adjoint function _norm_forward(μ, σ², x, γ, β, ϵ)
    N = ndims(x)

    σ²ϵ = σ² .+ ϵ
    inv_deno = 1 ./ sqrt.(σ²ϵ)
    res_1 = (x .- μ) .* inv_deno
    res_2 = γ .* res_1
    res = res_2 .+ β

    function norm_backward(Δ)
        reduce_dims_affine = filter(i -> isone(size(β, i)), 1:N)
        reduce_dims_stats = filter(i -> isone(size(σ², i)), 1:N)

        Δx = inv_deno .* Δ
        Δμ = -sum(Δx; dims=reduce_dims_stats)
        Δσ² = sum(-eltype(x)(0.5) .* res_2 .* Δ ./ σ²ϵ; dims=reduce_dims_stats)
        Δγ = sum(res_1 .* Δ; dims=reduce_dims_affine)
        Δβ = sum(Δ; dims=reduce_dims_affine)

        return (Δμ, Δσ², Δx, Δγ, Δβ, nothing)
    end

    return res, norm_backward
end

Zygote.@adjoint function _norm_forward(μ, σ², x, ϵ)
    N = ndims(x)

    σ²ϵ = σ² .+ ϵ
    inv_deno = 1 ./ sqrt.(σ²ϵ)
    res = (x .- μ) .* inv_deno

    function norm_backward(Δ)
        reduce_dims_stats = filter(i -> isone(size(σ², i)), 1:N)

        Δx = inv_deno .* Δ
        Δμ = -sum(Δx; dims=reduce_dims_stats)
        Δσ² = sum(-eltype(x)(0.5) .* res_2 .* Δ ./ σ²ϵ; dims=reduce_dims_stats)

        return (Δμ, Δσ², Δx, nothing)
    end

    return res, norm_backward
end
