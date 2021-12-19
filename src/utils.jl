function normalise01(x::AbstractArray)
    μ = mean(x)
    return (x .- μ) ./ (stdm(x, μ; corrected=false) .+ eps(eltype(x)))
end

Zygote.@adjoint function stdm(x, μ; corrected::Bool=true)
    σ = stdm(x, μ; corrected=corrected)
    function stdm_sensitivity(Δ)
        xμ = x .- μ
        ΔdivNσ = Δ ./ (σ .* (length(x) - corrected))
        return (xμ .* ΔdivNσ, zero(μ))
    end
    return σ, stdm_sensitivity
end
