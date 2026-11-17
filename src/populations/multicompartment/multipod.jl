# Multipod: an AdEx soma with N_d passive dendrites in a star (every dendrite attached to the soma).
#
# Purpose: the `Tripod` model generalised to any number of dendrites (1 to `MULTIPOD_MAX_DENDRITES`), with the same equations,
#          synapses, spike rules and integration schemes (`HeunScheme`, `PaperScheme`).
# Inputs:  `MultipodParameter` (morphology), `AdExParameter` (soma), synapse models, `PostSpike`, the scheme.
# Outputs: the population types `Multipod1`, ..., `Multipod8` (subtypes of `Multipod{D}`), their `integrate!` methods and the
#          `Multipod(; ...)` constructor that selects the type from the number of dendrites.
#
# Layout. One concrete struct per number of dendrites `D`, generated at module load with named fields `d1..dD`, `v_d1..v_dD`,
# `synvars_d1..`, `receptors_d1..` (the field access of `Tripod`). The integration code is written once for `Multipod{D}`: it
# reads the per-dendrite fields as tuples (`_dendrites`, `_v_dends`, ...), which the compiler unrolls for each `D`.
# The arithmetic of every step is the one of `Tripod` (same expressions, same order of the sums), so that `D = 2` reproduces
# `Tripod` bit for bit.

# ---------------------------------------------------------------------------------------------------------------------------------
# Abstract type and parameters
# ---------------------------------------------------------------------------------------------------------------------------------

"""
    Multipod{D} <: AbstractDendriteIF

Supertype of the `Multipod` populations with `D` dendrites; the concrete types are `Multipod1`, ..., `Multipod$(MULTIPOD_MAX_DENDRITES)`
(`SNNModels.Multipod3`, ...). Build them with `Multipod(; param = MultipodParameter(ds = [...]), ...)` or `Multipod{D}(; ...)`.
See `Multipod`.
"""
abstract type Multipod{D} <: AbstractDendriteIF end

"""
    MultipodParameter(; ds = [(200um, 400um), (200um, 400um), (200um, 400um)], physiology = human_dend, d = 4um,
                        El_dend = nothing, geometry = [(:s=>:d1), ..., (:s=>:dN)]) -> DendNeuronParameter

Morphology of a `Multipod` neuron: a soma and `N_d = length(ds)` dendrites (`1 <= N_d <= $(MULTIPOD_MAX_DENDRITES)`), named `d1`, ...,
`dN`, all attached to the soma.

# Keywords
- `ds`: one entry per dendrite: a length (cm, e.g. `300um`), a `(min, max)` range from which the length of each neuron is
  drawn, or `length => diameter` (a per-dendrite diameter, e.g. `300um => 3um`).
- `physiology::Physiology = human_dend`: cable constants `Ri`, `Rd`, `Cd`, common to all dendrites.
- `d = 4um`: dendrite diameter, a number (all dendrites) or a vector with one diameter per dendrite (then every entry of
  `ds` becomes `length => d[k]`, and `param.d` keeps the default 4 um, unused).
- `El_dend = nothing`: dendritic leak reversal (mV); `nothing`: the somatic `adex.El` at construction (see `Tripod`).
- `geometry`: the star `[(:s=>:d1), ..., (:s=>:dN)]`, the only topology implemented; any other value raises an error.

The cable constants of dendrite `k` are those of `create_dendrite` (the formulas of `Tripod`): with length `l` and diameter
`d`, ``g_{ax} = \\pi d^2 / (4 R_i l)``, ``g_m = \\pi d l / R_d``, ``C = C_d \\pi d l``. The returned `DendNeuronParameter` has
`type = MultipodNeuron()`, so that `Population(param)` builds a `Multipod` for any number of dendrites.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
param = SNN.MultipodParameter(ds = [400um, 250um => 3um, (150um, 300um)])
M = SNN.Multipod(N = 10, param = param)
```
"""
function MultipodParameter(;
    ds = [(200um, 400um), (200um, 400um), (200um, 400um)],
    physiology = human_dend,
    d = 4um,
    El_dend = nothing,
    geometry = _star_geometry(length(ds)),
)
    n = length(ds)
    1 <= n <= MULTIPOD_MAX_DENDRITES ||
        throw(ArgumentError("Multipod needs 1 to $(MULTIPOD_MAX_DENDRITES) dendrites, `ds` has $n entries"))

    # only the star is implemented: each dendrite once, attached to the soma
    Set(geometry) == Set(_star_geometry(n)) && length(geometry) == n ||
        throw(ArgumentError("Multipod implements the star geometry $(_star_geometry(n)) only, got $(geometry)"))

    # per-dendrite diameters: carried by the entries of `ds` as `length => diameter`
    if d isa AbstractVector
        length(d) == n || throw(ArgumentError("`d` has $(length(d)) diameters for $n dendrites"))
        any(l -> l isa Pair, ds) && throw(ArgumentError("give the diameters either in `ds` (length => diameter) or in `d`, not both"))
        ds = [ds[k] => d[k] for k in 1:n]
        d = 4um
    end

    return DendNeuronParameter(ds = collect(ds), physiology = physiology, d = d, El_dend = El_dend,
                               geometry = collect(geometry), type = MultipodNeuron())
end

# Length and diameter of dendrite `k` of `param`: a `length => diameter` entry overrides the common diameter `param.d`.
_dend_geometry(param::DendNeuronParameter, k::Int) =
    (entry = param.ds[k]; entry isa Pair ? (entry.first, entry.second) : (entry, param.d))

# The `Dendrite` `k` of a Multipod (the call `Tripod` makes for `d1`, `d2`, with the diameter of that dendrite).
function _multipod_dendrite(N, param::DendNeuronParameter, k::Int, adex, ::Type{FT}) where {FT}
    k <= length(param.ds) || throw(ArgumentError("`param.ds` has $(length(param.ds)) entries, the Multipod needs dendrite $k"))
    l, d = _dend_geometry(param, k)
    return create_dendrite(N, l; El = _dend_El(param, adex), FT = FT, physiology = param.physiology, d = d)
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Concrete types: one struct per number of dendrites, generated at load time
# ---------------------------------------------------------------------------------------------------------------------------------

# Name of the concrete type with `D` dendrites (`Multipod3`).
_multipod_typename(D::Int) = Symbol(:Multipod, D)

# Names of the per-dendrite fields with prefix `pre` (`:v_d` -> `v_d1`, ..., `v_dD`).
_dend_fieldnames(pre::Symbol, D::Int) = ntuple(k -> Symbol(pre, k), D)

# Struct definition of `Multipod<D>` as an expression for `@snn_kw`; the fields follow `Tripod` (same order, so that the same
# random draws initialise the same quantities), with `d1..dD` where `Tripod` has `d1`, `d2`.
function _multipod_struct_expr(D::Int)
    name = _multipod_typename(D)
    super = Symbol(:_MultipodSuper, D)          # `@snn_kw` takes the supertype as a name

    per_dend(pre, rhs) = [Expr(:(=), Expr(:(::), Symbol(pre, k), rhs[1]), rhs[2](k)) for k in 1:D]

    fields = Any[
        :(id::String = randstring(12)),
        :(name::String = "Multipod"),
        :(N::IT = 100),
        :(param::DendNeuronParameter = MultipodParameter(ds = fill((200um, 400um), $D))),
        :(adex::SOMAT = AdExParameter(FT = eltype(VFT))),
        :(soma_syn::SYNS = convert_precision(TripodSomaSynapse, eltype(VFT))),
        :(dend_syn::SYND = convert_precision(TripodDendSynapse, eltype(VFT))),
        :(spike::PST = PostSpike(FT = eltype(VFT))),
        :(Vspike::Float32 = 0mV),
        :(scheme::SCH = HeunScheme()),
        # dendrites: cable constants per neuron
        per_dend(:d, (:VDT, k -> :(_multipod_dendrite(N, param, $k, adex, eltype(VFT)))))...,
        # membrane potentials, adaptation and external currents
        :(v_s::VFT = rand_value(N, adex.Vt, adex.Vr)),
        :(w_s::VFT = zeros(N)),
        per_dend(:v_d, (:VFT, k -> :(rand_value(N, adex.Vt, adex.Vr))))...,
        :(I::VFT = zeros(N)),
        :(I_d::VFT = zeros(N)),
        # synapses: state variables and input buffers of each compartment
        :(synvars_s::SYNSV = synaptic_variables(soma_syn, N, eltype(VFT))),
        per_dend(:synvars_d, (:SYNDV, k -> :(synaptic_variables(dend_syn, N, eltype(VFT)))))...,
        :(receptors_s::RECTS = synaptic_receptors(soma_syn, N, eltype(VFT))),
        per_dend(:receptors_d, (:RECTD, k -> :(synaptic_receptors(dend_syn, N, eltype(VFT)))))...,
        # spike model and threshold
        :(fire::VBT = zeros(Bool, N)),
        :(spikes::SpikeIndex = owned_spike_index(N)),
        :(tabs::VFT = zeros(Int, N)),
        :(θ::VFT = ones(N) * adex.Vt),
        :(records::Dict = Dict()),
        # work arrays: derivatives (v_s, v_d1..v_dD, w_s), synaptic currents and stage potentials (s, d1..dD)
        :(Δv::MFT = zeros(N, $(D + 2))),
        :(Δv_temp::MFT = zeros(N, $(D + 2))),
        :(is::MFT = zeros(N, $(D + 1))),
        :(v_pred::MFT = zeros(N, $(D + 1))),
    ]

    params = Any[
        :(VFT = Vector{Float32}),
        :(MFT = Matrix{eltype(VFT)}),
        :(VDT = Dendrite{VFT}),
        :(SYNS <: AbstractSynapseParameter),
        :(SYND <: AbstractSynapseParameter),
        :(SYNSV <: AbstractSynapseVariable),
        :(SYNDV <: AbstractSynapseVariable),
        :(SOMAT <: AbstractGeneralizedIFParameter),
        :(PST <: AbstractSpikeParameter),
        :(RECTS <: NamedTuple),
        :(RECTD <: NamedTuple),
        :(SCH <: AbstractTripodScheme),
        :(IT = Int32),
    ]

    header = Expr(:(<:), Expr(:curly, name, params...), super)
    return super, Expr(:struct, false, header, Expr(:block, fields...))
end

# Tuple of the per-dendrite fields `pre1, ..., preD` of `p` (unrolled at compile time for each `D`).
@generated function _dend_fields(p::Multipod{D}, ::Val{pre}) where {D,pre}
    return Expr(:tuple, (:(getfield(p, $(QuoteNode(f)))) for f in _dend_fieldnames(pre, D))...)
end
_dendrites(p::Multipod) = _dend_fields(p, Val(:d))
_v_dends(p::Multipod) = _dend_fields(p, Val(:v_d))
_synvars_dends(p::Multipod) = _dend_fields(p, Val(:synvars_d))
_receptors_dends(p::Multipod) = _dend_fields(p, Val(:receptors_d))

"""
    ndendrites(p::Multipod) -> Int

Number of dendrites of a `Multipod` population.
"""
ndendrites(::Multipod{D}) where {D} = D

# The struct definitions: `Multipod1`, ..., `Multipod8`. Generated here, at module load, so that every session has the types
# (JLD2 rebuilds a saved population from the name of its type).
for D in 1:MULTIPOD_MAX_DENDRITES
    super, ex = _multipod_struct_expr(D)
    @eval const $super = Multipod{$D}
    @eval @snn_kw $ex
end

# The concrete type with `D` dendrites.
const _MULTIPOD_TYPES = Tuple(getfield(@__MODULE__, _multipod_typename(D)) for D in 1:MULTIPOD_MAX_DENDRITES)
_multipod_type(D::Integer) = 1 <= D <= MULTIPOD_MAX_DENDRITES ? _MULTIPOD_TYPES[D] :
    throw(ArgumentError("Multipod needs 1 to $(MULTIPOD_MAX_DENDRITES) dendrites, got $D"))

# ---------------------------------------------------------------------------------------------------------------------------------
# Constructor and documentation
# ---------------------------------------------------------------------------------------------------------------------------------

@doc raw"""
    Multipod(; N = 100, param = MultipodParameter(), adex = AdExParameter(),
               soma_syn = TripodSomaSynapse, dend_syn = TripodDendSynapse, spike = PostSpike(), scheme = HeunScheme(), ...)
    Multipod{D}(; ...)

Population of neurons with an AdEx-like soma and ``N_d`` passive dendrites `d1`, ..., `dN` (``1 \le N_d \le 8``), each
attached to the soma through its axial conductance (star topology). The number of dendrites is `length(param.ds)`; the
population is an object of the concrete type `Multipod<N_d>` (`SNNModels.Multipod3`, ...), a subtype of `Multipod{N_d}`.
`ndendrites(p)` returns ``N_d``.

With ``N_d = 2`` it is the `Tripod` model: same equations, synapses, spike rules and integration schemes, and the same
trajectory bit for bit (potentials, adaptation, threshold, refractory counters and spike times) for the same parameters and
initial state, with `HeunScheme` (`Tripod`) and `PaperScheme` (`TripodPaper`, see `MultipodPaper`).

Connections select the compartment with the fourth argument of `SpikingSynapse`: `SpikingSynapse(pre, M, :glu, :d3; conn)`,
targets `:s`, `:d1`, ..., `:dN`.

# Equations
Those of `Tripod` with the sum of the axial currents over the ``N_d`` dendrites. Soma (parameters from `adex`), and for each
dendrite ``k = 1, \dots, N_d``:
```math
\begin{aligned}
C \frac{dV_s}{dt} &= g_L (E_L - V_s) + g_L \Delta_T\, e^{(V_s - \theta)/\Delta_T} - w_s
    - I_{syn,s} - \sum_{k=1}^{N_d} g_{ax,k}\,(V_s - V_{d,k}) + I \\
C_{d,k} \frac{dV_{d,k}}{dt} &= g_{m,k} (E_{L,k} - V_{d,k}) - I_{syn,d,k}
    + g_{ax,k}\,(V_s - V_{d,k}) + I_d \\
\tau_w \frac{dw_s}{dt} &= a (V_s - E_L) - w_s \\
\tau_A \frac{d\theta}{dt} &= V_t - \theta
\end{aligned}
```
The axial currents are summed in the order ``k = 1, \dots, N_d``. ``C_{d,k}``, ``g_{m,k}``, ``g_{ax,k}`` come from the length
and diameter of dendrite ``k`` (`create_dendrite`: ``g_{ax} = \pi d^2 / (4 R_i l)``, ``g_m = \pi d l / R_d``,
``C = C_d \pi d l``); ``E_{L,k}`` is `param.El_dend`, or `adex.El` at construction when it is `nothing`. The synaptic currents
are clamped to ``\pm 1500`` pA. ``I_d`` is injected into every dendrite. The exponential term is ``+g_L \Delta_T e^{(V_s -
\theta)/\Delta_T}``, as in `Tripod` and the AdEx model.

Spike, backpropagation and refractory period, and the integration: as in `Tripod` (see its Spike and Integration sections) for
`scheme = HeunScheme()`, and as in `PaperScheme` for `scheme = PaperScheme()`, where the axial current of each dendrite is frozen
over the step and the soma takes its two-stage step with the sum of the frozen axial currents.

# Fields
The fields of `Tripod`, with one field per dendrite where `Tripod` has `d1`, `d2`:
- `id`, `name = "Multipod"`, `N`, `records`; `param::DendNeuronParameter = MultipodParameter(...)` (default: ``N_d`` dendrites
  of length drawn in 200-400 um); `adex`, `soma_syn`, `dend_syn`, `spike`, `Vspike = 0mV`, `scheme = HeunScheme()`.
- `d1`, ..., `dN::Dendrite`: cable constants of each dendrite.
- `v_s`, `v_d1`, ..., `v_dN`: compartment potentials (mV); `w_s`: adaptation current (pA); `I`: somatic current; `I_d`:
  current injected into each dendrite (pA); `fire`, `spikes`, `tabs`, `θ`.
- `synvars_s`, `synvars_d1`, ...: synaptic state variables; `receptors_s`, `receptors_d1`, ...: input buffers.
- `Δv`, `Δv_temp` (`N x (N_d + 2)`: `v_s`, `v_d1..v_dN`, `w_s`), `is`, `v_pred` (`N x (N_d + 1)`): work arrays.
`VFT = Vector{Float64}` gives a `Float64` population, as for `Tripod`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
M = SNN.Multipod(N = 10, param = SNN.MultipodParameter(ds = [400um, 300um, 150um, 150um]))
E = SNN.Poisson(N = 50, param = SNN.PoissonParameter(20Hz))
syn = SNN.SpikingSynapse(E, M, :glu, :d3; conn = (p = 0.2, μ = 2.0))
model = SNN.compose(; E, M, syn)
SNN.monitor!(M, [:v_s, :v_d3, :fire])
SNN.sim!(; model, duration = 200ms)
```
"""
function Multipod(; param::DendNeuronParameter = MultipodParameter(), kwargs...)
    return _multipod_type(length(param.ds))(; param, kwargs...)
end

# `Multipod{D}(; ...)`: the type with `D` dendrites; the default `param` has `D` dendrites of length drawn in 200-400 um.
(::Type{Multipod{D}})(; kwargs...) where {D} = _multipod_type(D)(; kwargs...)

# one precision per population, as `Tripod`; the number of dendrites of `param` must be the one of the type
function snn_kw_finalize(::Type{T}, f::NamedTuple) where {D,T<:Multipod{D}}
    length(f.param.ds) == D ||
        throw(ArgumentError("$(nameof(T)) has $D dendrites, `param.ds` has $(length(f.param.ds)) entries"))
    _check_precision("Multipod", _state_float(T, :v_s), "adex" => f.adex, "soma_syn" => f.soma_syn, "dend_syn" => f.dend_syn,
                     "spike" => f.spike)
    f.scheme isa PaperScheme && !(f.soma_syn isa ReceptorSynapse && f.dend_syn isa ReceptorSynapse) &&
        throw(ArgumentError("Multipod with PaperScheme needs `ReceptorSynapse` models for `soma_syn` and `dend_syn`"))
    return f
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Integration: HeunScheme (the scheme of `Tripod`, see `tripod.jl`)
# ---------------------------------------------------------------------------------------------------------------------------------

"""
    integrate!(p::Multipod, param::DendNeuronParameter, dt::AbstractFloat)

Advance a `Multipod` population by `dt`: update the synapses of all compartments, then integrate the membrane equations with
the scheme of the population (`HeunScheme` or `PaperScheme`) and handle spikes and refractoriness (see `Multipod`).
"""
function integrate!(p::Multipod{D}, param::DendNeuronParameter, dt_::AbstractFloat) where {D}
    dt = _step_dt(p, dt_)   # in the precision of the population
    N = Int(p.N)
    @unpack spike, soma_syn, dend_syn = p
    @unpack τabs, up = spike

    # refractory counters in steps; each period lasts at least one step
    n_abs = max(1, round(Int, τabs / dt))
    n_ref = max(1, round(Int, up / dt)) + n_abs

    _check_state_lengths(N, p.v_s, p.w_s, p.fire, p.θ, p.tabs, p.I, p.I_d, _v_dends(p)...)
    for dend in _dendrites(p)
        _check_state_lengths(N, dend.gax, dend.gm, dend.El, dend.C)
    end
    (size(p.Δv, 1) >= N && size(p.Δv, 2) >= D + 2 && size(p.Δv_temp, 1) >= N && size(p.Δv_temp, 2) >= D + 2) ||
        throw(DimensionMismatch("Δv and Δv_temp must have N rows and $(D + 2) columns"))
    (size(p.is, 1) >= N && size(p.is, 2) >= D + 1 && size(p.v_pred, 1) >= N && size(p.v_pred, 2) >= D + 1) ||
        throw(DimensionMismatch("is and v_pred must have N rows and $(D + 1) columns"))

    if _chunk_synapse(soma_syn) && _chunk_synapse(dend_syn)
        # all stages are per-neuron with no shared scratch: one fork/join per population and step
        foreach_chunk_c(N) do c, rng
            _multipod_chunk!(p.scheme, p, dt, rng, n_abs, n_ref)
        end
    elseif !(p.scheme isa HeunScheme)
        throw(ArgumentError("Multipod with $(nameof(typeof(p.scheme))) needs receptor synapse models with chunk methods (ReceptorSynapse)"))
    else
        # synapse models without chunk methods: the stages one after the other over the whole population
        _multipod_serial!(p, dt, n_abs, n_ref)
    end
end

# one chunk with the Heun scheme: synapses, the two Heun stages and the state update of the neurons `rng`
function _multipod_chunk!(::HeunScheme, p::Multipod{D}, dt::AbstractFloat, rng::UnitRange{Int}, n_abs::Int, n_ref::Int) where {D}
    @unpack Δv, Δv_temp, fire, soma_syn, dend_syn = p

    # advance the conductances of every compartment
    update_synapses!(p, soma_syn, p.receptors_s, p.synvars_s, dt, rng)
    for (receptors, synvars) in zip(_receptors_dends(p), _synvars_dends(p))
        update_synapses!(p, dend_syn, receptors, synvars, dt, rng)
    end

    # first stage at the current state (Δv = 0), soma clamped where refractory
    @inbounds for k in 1:(D + 2)
        @simd ivdep for i in rng
            Δv[i, k] = zero(eltype(Δv))
            Δv_temp[i, k] = zero(eltype(Δv_temp))
        end
    end
    @inbounds for i in rng
        fire[i] = false
    end
    _tripod_clamp_soma!(p, rng, n_abs)
    _multipod_stage!(p, Δv, dt, rng)

    # second stage at the Euler-predicted state; a clamped soma does not move
    @inbounds for k in 1:(D + 2)
        @simd ivdep for i in rng
            Δv_temp[i, k] = Δv[i, k]
        end
    end
    _tripod_hold_soma!(p, rng, n_abs)
    _multipod_stage!(p, Δv, dt, rng)

    _multipod_state!(p, dt, rng, n_abs, n_ref)
end

# the same stages over the whole population, for synapse models that have no chunk methods
function _multipod_serial!(p::Multipod, dt::AbstractFloat, n_abs::Int, n_ref::Int)
    @unpack Δv, Δv_temp, fire, soma_syn, dend_syn = p
    update_synapses!(p, soma_syn, p.receptors_s, p.synvars_s, dt)
    for (receptors, synvars) in zip(_receptors_dends(p), _synvars_dends(p))
        update_synapses!(p, dend_syn, receptors, synvars, dt)
    end
    fill!(Δv, zero(eltype(Δv)))
    fill!(Δv_temp, zero(eltype(Δv_temp)))
    fill!(fire, false)
    rng = 1:Int(p.N)
    _tripod_clamp_soma!(p, rng, n_abs)
    _multipod_stage!(p, Δv, dt, rng)
    Δv_temp .= Δv
    _tripod_hold_soma!(p, rng, n_abs)
    _multipod_stage!(p, Δv, dt, rng)
    _multipod_state!(p, dt, rng, n_abs, n_ref)
end

# Left-to-right sum `((x + t[1]) + t[2]) + ...`, the order of the axial currents in the somatic equation of `Tripod`.
@inline _sum_left(x, t::Tuple) = _sum_left(x + first(t), Base.tail(t))
@inline _sum_left(x, ::Tuple{}) = x

# One Heun stage for the neurons `rng`: derivatives at the predicted state x + dt Δv (Δv = 0 at the first stage).
function _multipod_stage!(p::Multipod{D}, Δv::Matrix{FT}, dt::FT, rng::UnitRange{Int}) where {D,FT<:AbstractFloat}
    @unpack v_s, I_d, I, w_s, θ, is, v_pred = p
    @unpack adex, soma_syn, dend_syn, synvars_s = p
    @unpack C, gl, El, ΔT, a, τw = adex
    v_ds = _v_dends(p)
    dends = _dendrites(p)
    gax = map(d -> d.gax, dends)
    gm = map(d -> d.gm, dends)
    Eld = map(d -> d.El, dends)
    Cd = map(d -> d.C, dends)
    last(rng) <= p.N || throw(BoundsError(v_s, last(rng)))

    # stage potentials of the soma (column 1) and of the dendrites (columns 2..D+1)
    @inbounds @simd ivdep for i ∈ rng
        v_pred[i, 1] = v_s[i] + Δv[i, 1] * dt
    end
    for k in 1:D
        v_d = v_ds[k]
        @inbounds @simd ivdep for i ∈ rng
            v_pred[i, k + 1] = v_d[i] + Δv[i, k + 1] * dt
        end
    end

    # synaptic currents at the stage potentials, clamped to ±1500 pA
    @views _synaptic_current!(p, soma_syn, synvars_s, v_pred[:, 1], is[:, 1], rng)
    for (k, synvars) in enumerate(_synvars_dends(p))
        @views _synaptic_current!(p, dend_syn, synvars, v_pred[:, k + 1], is[:, k + 1], rng)
    end
    @inbounds for k in 1:(D + 1)
        @simd ivdep for i in rng
            is[i, k] = clamp(is[i, k], FT(-1500), FT(1500))
        end
    end

    # derivatives: the expressions and the `muladd` order of `Tripod` (bit-identical for D = 2; see `update_neuron!(::Tripod)`)
    @inbounds @simd ivdep for i ∈ rng
        vs = v_pred[i, 1]
        ws = muladd(Δv[i, D + 2], dt, w_s[i])

        # dendrites: leak - synapses + axial + external; the axial currents soma -> dendrite are kept for the soma
        ics = ntuple(Val(D)) do k
            vd = v_pred[i, k + 1]
            ic = (vs - vd) * gax[k][i]
            Δv[i, k + 1] = ((muladd(Eld[k][i] - vd, gm[k][i], -is[i, k + 1]) + ic) + I_d[i]) / Cd[k][i]
            ic
        end

        # soma: gl * (El - vs + ΔT * exp((vs - θ) / ΔT)) + I - (adaptation + synapses + axial currents)
        e = exp256((vs - θ[i]) / ΔT)
        Δv[i, 1] = muladd(gl, muladd(ΔT, e, El - vs), I[i] - _sum_left(ws + is[i, 1], ics)) / C
        Δv[i, D + 2] = muladd(a, vs - El, -ws) / τw
    end
end

# State update after the Heun stages (spike detection, refractoriness, clamping), neurons `rng`; the rules of `_tripod_state!`.
function _multipod_state!(p::Multipod{D}, dt::AbstractFloat, rng::UnitRange{Int}, n_abs::Int, n_ref::Int) where {D}
    @unpack v_s, w_s, fire, θ, tabs, Δv, Δv_temp = p
    @unpack spike, adex, Vspike = p
    @unpack AP_membrane, At, τA = spike
    @unpack Vr, Vt, b = adex
    v_ds = _v_dends(p)
    nabs_f = convert(eltype(tabs), n_abs)    # the counters are floats: compare and store without mixing types
    nref_f = convert(eltype(tabs), n_ref)
    FT = eltype(v_s)

    # Branchless, as in `_tripod_state!`: every case (backpropagation, refractory, free integration, spike) is evaluated and the
    # applicable one kept.
    @inbounds @simd ivdep for i ∈ rng
        t1 = tabs[i] - 1
        th1 = θ[i] + dt * (Vt - θ[i]) / τA
        bp = t1 > nabs_f                 # backpropagation period
        rf = !bp & (t1 > 0)              # absolute refractory period
        act = t1 <= 0                    # free integration (or spike)
        vs0 = v_s[i]
        ws0 = w_s[i]
        dv1 = Δv[i, 1]

        # bp or rf: the soma is clamped (the Heun stages already integrated the dendrites against the clamped soma)
        vsc = ifelse(bp, AP_membrane, Vr)

        # act: spike or Heun update
        spk = act & (vs0 + dv1 * dt >= Vspike)
        keep = act & !spk
        vs_h = convert(FT, vs0 + 0.5 * dt * (Δv_temp[i, 1] + dv1))   # the Float64 sum stored in the state precision
        ws_h = convert(FT, ws0 + 0.5 * dt * (Δv_temp[i, D + 2] + Δv[i, D + 2]))

        clamped = bp | rf
        fire[i] = spk
        Δv[i, 1] = ifelse(spk, AP_membrane - vs0, dv1)
        v_s[i] = ifelse(clamped, vsc, ifelse(spk, AP_membrane, ifelse(keep, vs_h, vs0)))

        # dendrites: Heun update, also while the soma is clamped; they hold on the spike step
        ntuple(Val(D)) do k
            vd = v_ds[k][i]
            vd_h = convert(FT, vd + 0.5 * dt * (Δv_temp[i, k + 1] + Δv[i, k + 1]))
            v_ds[k][i] = ifelse(spk, vd, vd_h)
            nothing
        end

        w_s[i] = ifelse(spk, ws0 + b, ifelse(keep, ws_h, ws0))
        θ[i] = ifelse(spk, th1 + At, th1)
        tabs[i] = ifelse(spk, nref_f, t1)
    end
end

export Multipod, MultipodParameter, MultipodNeuron, ndendrites
