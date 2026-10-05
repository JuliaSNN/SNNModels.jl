# Membrane parameters of AdExParameter and IFParameter (also used by Tripod and BallAndStick
# through their `adex` field). The four fields C, gl, R, τm are tied by
#     τm = C / gl,    R = 1nS / gl,
# so a pair of independent values fixes all four. The models read different members of the
# quadruple (AdEx and IF integrate with τm and R, Tripod and BallAndStick with C and gl), so the
# four stored values must always agree.

const MEMBRANE_FIELDS = (:C, :gl, :R, :τm)

"Parameter types whose `C`, `gl`, `R`, `τm` follow the membrane pair rule."
const MembraneParameter = Union{AdExParameter,IFParameter}

_given(x) = !all(isnan, x)
_agree(a, b) = all(isapprox.(a, b; rtol = 1.0f-3))

@doc raw"""
    resolve_membrane(C, gl, R, τm; default, name = "") -> (; C, gl, R, τm)

Complete the membrane quadruple from the values that were given (a value is "not given" when it
is `NaN`, the field default of `AdExParameter` and `IFParameter`). With
``\tau_m = C / g_l`` and ``R = 1\,\mathrm{nS} / g_l``:

- nothing given: the type's default pair `default` is used (AdEx: `C = 281pF, gl = 40nS`;
  IF: `τm = 15ms, R = 0.06`);
- exactly one independent value given: `ArgumentError`. A default is never combined with a given
  value, so a single value does not determine the others;
- two independent values given, any of (C, gl), (C, R), (C, τm), (gl, τm), (R, τm): the other two
  are derived. `gl` and `R` are the same quantity; if both are given they must agree;
- three or more given: accepted when they are consistent (relative tolerance 1e-3, so that values rounded to four significant digits pass), otherwise
  `ArgumentError`. This is the case when an existing parameter is rebuilt with all its fields.

Values can be scalars or vectors (heterogeneous parameters); the relations are applied element-wise.
"""
function resolve_membrane(C, gl, R, τm; default::NamedTuple, name = "")
    where_ = isempty(name) ? "" : "$name: "
    gC, ggl, gR, gτ = _given(C), _given(gl), _given(R), _given(τm)
    if !(gC || ggl || gR || gτ)
        d = merge((C = NaN32, gl = NaN32, R = NaN32, τm = NaN32), default)
        return resolve_membrane(d.C, d.gl, d.R, d.τm; default, name)
    end
    if gR
        gl_R = nS ./ R
        if ggl
            _agree(gl, gl_R) || throw(ArgumentError(
                "$(where_)gl = $gl and R = $R disagree (R must be 1nS / gl)",
            ))
        else
            gl, ggl = gl_R, true
        end
    end
    n = gC + ggl + gτ
    if n == 1
        given = join([string(k) for (k, g) in zip(MEMBRANE_FIELDS, (gC, ggl, gR, gτ)) if g], ", ")
        throw(ArgumentError(
            "$(where_)membrane parameters must be given as a pair, got only $given. " *
            "Give two of C, gl (or R), τm, or none to use the default pair $default; " *
            "a given value is never combined with a default.",
        ))
    end
    if !gC
        C = τm .* gl
    elseif !ggl
        gl = C ./ τm
    elseif gτ
        _agree(τm, C ./ gl) || throw(ArgumentError(
            "$(where_)C = $C, gl = $(gl), τm = $τm are inconsistent (τm must be C / gl)",
        ))
    end
    τm = gτ ? τm : C ./ gl
    R = gR ? R : nS ./ gl
    return (; C, gl, R, τm)
end

@doc raw"""
    membrane_update(current::NamedTuple, new::NamedTuple) -> (; C, gl, R, τm)

Membrane quadruple after changing some of its members. `current` holds the four consistent
values, `new` the changed ones (keys among `C, gl, R, τm`).

- If `new` alone determines a pair (two independent values, see [`resolve_membrane`](@ref)), the
  result is resolved from `new` only.
- If `new` changes one quantity, the others follow a fixed rule:
  - `τm`: `gl` (and `R`) are kept, `C = τm * gl`;
  - `C`: `gl` (and `R`) are kept, `τm = C / gl`;
  - `gl` or `R`: `C` is kept, `τm = C / gl` and `R = 1nS / gl` (or `gl = 1nS / R`).

So changing `τm` changes the capacitance, the quantity Tripod and BallAndStick integrate with.
"""
function membrane_update(current::NamedTuple, new::NamedTuple)
    m = _membrane_update(current, new)
    # a scalar given for a heterogeneous parameter applies to every neuron
    return map(v -> current.C isa AbstractVector && !(v isa AbstractVector) ? fill(v, length(current.C)) : v, m)
end

function _membrane_update(current::NamedTuple, new::NamedTuple)
    for k in keys(new)
        k in MEMBRANE_FIELDS || throw(ArgumentError("$k is not a membrane parameter"))
    end
    nan = (C = NaN32, gl = NaN32, R = NaN32, τm = NaN32)
    n = haskey(new, :C) + (haskey(new, :gl) || haskey(new, :R)) + haskey(new, :τm)
    if n >= 2
        v = merge(nan, new)
        return resolve_membrane(v.C, v.gl, v.R, v.τm; default = (;))
    elseif haskey(new, :τm)
        return (; C = new.τm .* current.gl, current.gl, current.R, new.τm)
    elseif haskey(new, :C)
        return (; new.C, current.gl, current.R, τm = new.C ./ current.gl)
    elseif haskey(new, :gl) || haskey(new, :R)
        v = merge(nan, (; C = current.C), new)
        return resolve_membrane(v.C, v.gl, v.R, v.τm; default = (;))
    end
    return (; current.C, current.gl, current.R, current.τm)
end

_membrane(p) = (; C = p.C, gl = p.gl, R = p.R, τm = p.τm)

snn_kw_finalize(::Type{<:AdExParameter}, f::NamedTuple) = merge(
    f,
    resolve_membrane(f.C, f.gl, f.R, f.τm; default = (C = 281pF, gl = 40nS), name = "AdExParameter"),
)
snn_kw_finalize(::Type{<:IFParameter}, f::NamedTuple) = merge(
    f,
    resolve_membrane(f.C, f.gl, f.R, f.τm; default = (τm = 15ms, R = 0.06f0), name = "IFParameter"),
)

"""
    with_membrane(p; C, gl, R, τm)

Copy of the parameter `p` (`AdExParameter` or `IFParameter`) with membrane parameters changed by
[`membrane_update`](@ref): one value follows the update rule, two independent values define the
new pair. `@update!` on one of these fields and property assignment on the mutable
`AdExParameter` use the same rule.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
p = SNN.AdExParameter()                       # C = 281pF, gl = 40nS, τm = 7.025ms
q = SNN.with_membrane(p; τm = 20ms)           # gl kept: C = 800pF
r = SNN.with_membrane(p; τm = 20ms, C = 281pF) # new pair: gl = 14.05nS
```
"""
function with_membrane(p::T; kwargs...) where {T<:MembraneParameter}
    m = membrane_update(_membrane(p), NamedTuple(kwargs))
    fields = (; [f => getfield(p, f) for f in fieldnames(T)]...)
    return T.name.wrapper(; merge(fields, m)..., FT = T.parameters[1])
end

function update_with_merge(base::MembraneParameter, path::Vector{Symbol}, value, full_path = nothing)
    if length(path) == 1 && path[1] in MEMBRANE_FIELDS
        return with_membrane(base; NamedTuple{(path[1],)}((value,))...)
    end
    return invoke(update_with_merge, Tuple{Any,Vector{Symbol},Any,Any}, base, path, value, full_path)
end

function Base.setproperty!(p::AdExParameter, f::Symbol, v)
    if f in MEMBRANE_FIELDS
        m = membrane_update(_membrane(p), NamedTuple{(f,)}((v,)))
        for g in MEMBRANE_FIELDS
            setfield!(p, g, convert(fieldtype(typeof(p), g), getfield(m, g)))
        end
        return v
    end
    return setfield!(p, f, convert(fieldtype(typeof(p), f), v))
end

export resolve_membrane, membrane_update, with_membrane
