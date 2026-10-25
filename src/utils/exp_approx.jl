# Exponential approximations selectable in the receptor models (NMDA block, conductance decay) and the original-code scheme of
# `Tripod`. Loaded before the populations: the predefined receptors are built at load time. `exp64` and `exp256` are in util.jl.

"""
    exp32o(x::AbstractFloat)

`exp32` of the published Tripod code (`equations.jl` of `aquaresima/tripod_neuron`): ``(1 + x/32)^{32}`` with five squarings,
inputs below `-10` replaced by `-32` so that the result is exactly `0` (`exp32` of this package clamps them to `-10`, giving
about `3e-6`). Used by the original-code options (`NMDAVoltageDependency(approx = 32)`, `ReceptorSynapse(decay = 32)`,
`PaperScheme`). Not exported.
"""
@inline function exp32o(x::T) where {T<:AbstractFloat}
    x = ifelse(x < T(-10), T(-32), x)
    x = one(T) + x / T(32)
    x *= x
    x *= x
    x *= x
    x *= x
    x *= x
    return x
end

# `_expapprox(Val(n), x)`: the exponential used by the options `approx` (NMDA block) and `decay` (receptor decay): `n` is the number
# of the power-law approximation `(1 + x/n)^n` (32: `exp32o`, 64: `exp64`, 256: `exp256`), 0 the exact `exp`.
const _EXP_APPROX = (0, 32, 64, 256)
_check_exp_approx(what, n) = n in _EXP_APPROX ||
    throw(ArgumentError("$what must be one of $(_EXP_APPROX) (0 = exact exp, n = (1 + x/n)^n), got $n"))
@inline _expapprox(::Val{0}, x) = exp(x)
@inline _expapprox(::Val{32}, x) = exp32o(x)
@inline _expapprox(::Val{64}, x) = exp64(x)
@inline _expapprox(::Val{256}, x) = exp256(x)
