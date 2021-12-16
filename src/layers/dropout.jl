const allow_variational_hidden_dropout_mask_reset = Ref(false)

@inline is_variational_hidden_dropout_mask_reset_allowed() = allow_variational_hidden_dropout_mask_reset[]

@inline function _update_is_variational_hidden_dropout_mask_reset_allowed(val::Bool)
    return allow_variational_hidden_dropout_mask_reset[] = val
end

"""
    update_is_variational_hidden_dropout_mask_reset_allowed()
    update_is_variational_hidden_dropout_mask_reset_allowed(val::Bool)

Allow/Disallows variational_hidden_dropout mask reset.
"""
@inline function update_is_variational_hidden_dropout_mask_reset_allowed(val::Bool=!is_variational_hidden_dropout_mask_reset_allowed())
    return Zygote.hook(Δ -> begin
                           _update_is_variational_hidden_dropout_mask_reset_allowed(!val)
                           return Δ
                       end, _update_is_variational_hidden_dropout_mask_reset_allowed(val))
end

mutable struct VariationalHiddenDropout{F,M}
    p::F
    mask::M
    active::Union{Bool,Nothing}
end

Flux.trainable(::VariationalHiddenDropout) = ()

Flux.@functor VariationalHiddenDropout

"""
    VariationalHiddenDropout(p, s)

Like `Dropout` but with a mask which is not reset after each function call.
This is important for models like DeepEquilibriumModels where mask is reset
only after the solve call is complete.

# Arguments

* `p`: Probability of dropout.
* `s`: Size of the Input. Pass `1` as the last value of the tuple

See also [`update_is_variational_hidden_dropout_mask_reset_allowed`](@ref)
"""
function VariationalHiddenDropout(p, s)
    @assert 0 ≤ p ≤ 1
    mask = zeros(Float32, s)
    vd = VariationalHiddenDropout(p, mask, nothing)
    reset_mask!(vd)
    return vd
end

function reset_mask!(a::VariationalHiddenDropout)
    Flux.rand!(a.mask)
    return a.mask .= Flux._dropout_kernel.(a.mask, a.p, 1 - a.p)
end

Zygote.@nograd reset_mask!

function (a::VariationalHiddenDropout)(x)
    Flux._isactive(a) || return x
    is_variational_hidden_dropout_mask_reset_allowed() && reset_mask!(a)
    return variational_hidden_dropout(x, a.mask; active=true)
end

function Flux.testmode!(m::VariationalHiddenDropout, mode=true)
    return (m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode; m)
end

function Base.show(io::IO, d::VariationalHiddenDropout)
    print(io, "VariationalDropout(", d.p)
    print(io, ", size = $(repr(size(d.mask)))")
    return print(io, ")")
end

function variational_hidden_dropout(x, mask; active::Bool=true)
    active || return x
    return x .* mask
end

Zygote.@adjoint function variational_hidden_dropout(x, mask; active::Bool=true)
    active || return x, Δ -> (Δ, nothing)
    return x .* mask, Δ -> (Δ .* mask, nothing)
end
