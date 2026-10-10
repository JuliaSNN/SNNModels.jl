#=
Voltage-based STDP (Clopath et al. 2010), two implementations of the same rule:

  vSTDPParameter / vSTDPVariables              event-driven, the default. The weights are updated lazily.
  vSTDPDenseParameter / vSTDPDenseVariables    clock-driven reference, every synapse at every step.

Both are generic over the floating-point type of the synapse (`float_type`): the state follows the precision of
the postsynaptic population the synapse is built on.
=#

@doc raw"""
    vSTDPParameter(; A_LTD = 8e-4, A_LTP = 1.4e-3, θ_LTD = -70mV, θ_LTP = -49mV,
                     τu = 20ms, τv = 7ms, τx = 15ms, Wmax = 30pF, Wmin = 0.1pF, epoch = 10)

Voltage-based STDP in the form of Clopath et al. (2010), event-driven: the default form of the rule.
Depression is triggered by a presynaptic spike and depends on a low-pass filtered postsynaptic membrane
potential; potentiation depends on a presynaptic spike trace, on the instantaneous postsynaptic potential and on
a second low-pass filter of it. It is a subtype of `LTPParameter`; pass it as `LTPParam` to a `SpikingSynapse`
that targets a population with a membrane potential (`v_post` of the synapse). Its state is `vSTDPVariables`.
The clock-driven form that evaluates the potentiation on every synapse at every step is
[`vSTDPDenseParameter`](@ref) (the reference of this rule; up to 1.9 `vSTDPParameter` was that rule).

!!! warning "Changed in SNNModels 1.10: `vSTDPParameter` is event-driven and `syn.W` is lazy"
    Up to 1.9 `vSTDPParameter` was the dense rule (now `vSTDPDenseParameter`, same equations, same fields without
    `epoch`). The event-driven rule gives the same weights (to rounding, see below) at a cost that does not
    grow with the number of synapses, but it materialises the weights lazily: `syn.W` holds the weights as of
    the last materialisation of each synapse, NOT the current weights. Read them with
    `plasticity_value(syn, :W)` (what `monitor!(syn, [:W])` does), or call `sync_traces!(syn)` first and read
    `syn.W`. `forward!` is always correct. See "Reading the weights".

# Equations
With ``V_i`` the postsynaptic potential (`v_post`), ``u_i``, ``v_i`` its low-pass filters,
``x_j`` the presynaptic trace and ``[z]_+ = \max(z, 0)``:
```math
\begin{aligned}
τ_x \frac{dx_j}{dt} &= -x_j + \sum_k δ(t - t_j^k), \qquad
τ_u \frac{du_i}{dt} = -u_i + V_i, \qquad
τ_v \frac{dv_i}{dt} = -v_i + V_i,\\
Δw_{ij}^{LTD} &= -A_{LTD}\,[u_i - θ_{LTD}]_+ \quad \text{at each presynaptic spike of } j,\\
\frac{dw_{ij}^{LTP}}{dt} &= A_{LTP}\, x_j\, [v_i - θ_{LTD}]_+\, [V_i - θ_{LTP}]_+ .
\end{aligned}
```
A presynaptic spike increases ``x_j`` by ``1/τ_x`` (units 1/ms) and potentiation is a rate, integrated over
`dt`; the rule is independent of `dt`, as in Clopath et al. (2010). Note that both thresholds of the LTP term are
as in the code: the filtered potential ``v`` is compared with `θ_LTD`, the instantaneous one with `θ_LTP`.
Per step (the order of the dense rule): (1) forward Euler for ``x`` (this step's spikes included), (2) forward Euler
for ``u`` and ``v`` (driven by the current `v_post`), (3) LTD at the presynaptic spikes of the step, clamped at
`Wmin`, then (4) the LTP increment `dt * A_LTP * x * ...` on all outgoing synapses, clamped at `Wmax`. At the first
call the voltage traces ``u`` and ``v`` are set to the current postsynaptic potential (they are not started at 0 mV).

# What is exact and what is paid
The filters ``u_i``, ``v_i`` of the postsynaptic potential and the thresholded drive
``g_i(n) = [v_i(n) - θ_{LTD}]_+ [V_i(n) - θ_{LTP}]_+`` are properties of the postsynaptic neuron,
so they are still computed at every step (cost O(`Npost`)). The presynaptic trace obeys
``x_j(n) = q\,x_j(n-1) + s_j(n)/τ_x`` with ``q = 1 - dt/τ_x``, hence between presynaptic spikes
``x_j(n) = a_j q^n`` with ``a_j`` constant (a spike adds ``q^{-n}/τ_x`` to it). The LTP weight
``dt\,A_{LTP} \sum_n x_j(n) g_i(n)`` is then ``dt\,A_{LTP}\,a_j (K_i(n) - K_i(n_k - 1))`` since the
last presynaptic spike ``n_k`` of ``j``, where ``K_i(n) = \sum_{m \le n} q^m g_i(m)`` is accumulated
per postsynaptic neuron. Per synapse only a snapshot of ``K_i`` taken at the previous presynaptic
spike of ``j`` is kept. The weights are materialised

- at a presynaptic spike of ``j`` (in `update_traces!`, before `forward!`, so that the spike is
  transmitted with the current weight; the LTD of the spike then acts on the materialised weight),
  for the outgoing synapses of ``j`` only;
- for all synapses once per epoch (`epoch` times `τx`, default 10 `τx`): `K` is a sum with
  geometrically decaying terms and the difference ``K(n) - K(n_k - 1)`` loses all its digits after
  about 35 `τx` in `Float64`, so the accumulator, the snapshots and the amplitudes ``a_j`` are
  restarted every epoch. With `epoch = 10` the relative error of an increment is about
  ``10^{-16} e^{10} = 2 \cdot 10^{-12}``;
- on demand (`sync_traces!`, a change of `dt`, `τx`, `A_LTP` or `Wmax`).

`K`, the snapshots and `a_j` are `Float64` whatever the precision of the synapse (8 B per synapse for the
snapshot): with `Float32` the same cancellation would give a relative error of ``6 \cdot 10^{-8} e^{10} = 10^{-3}``
of an increment at the end of an epoch. The weights, the filters and the thresholds are in the precision of the
synapse (`float_type`). Against the dense rule in `Float64` the final weights agree to about 1e-12 (see the validation
README); in `Float32` the difference is the `Float32` accumulation error of the dense rule (1e-5 relative).

# Bounds
LTD acts at presynaptic spikes, before the LTP of the same step, on the materialised weight,
and is clamped at `Wmin` there. The LTP increments are non-negative (`A_LTP >= 0` is enforced),
so clamping at `Wmax` once after the accumulation equals clamping at every step
(``\min(W_{max}, \min(W_{max}, a) + b) = \min(W_{max}, a + b)`` for ``b \ge 0``), including the
case where LTD has pushed the weight to `Wmin`: the clamp at `Wmin` is applied at the spike, the
accumulation afterwards only adds. The one difference: a weight that is initially above `Wmax` is
clamped by the dense rule at the first step and by this rule at the first materialisation (the
recorded and forwarded weights are the clamped ones in both).

# Reading the weights
`syn.W` holds the weights as of the last materialisation of each synapse, NOT the current weights. The
current weights are given by `plasticity_value(syn, :W)` (a scratch vector, state unchanged; used by `monitor!` of
`:W`, see [`plasticity_value`](@ref)), the same accessor as for the lazily decayed traces. Or call
[`sync_traces!`](@ref)`(syn)` (materialise all, in place, restarts the accumulation epoch) before reading `syn.W`
directly (plots, `matrix`, saving a model, another rule or a metaplasticity object that rescales or rewires the
weights: pending increments are not rescaled by a rule that acts on `W` in between). The library's own copy and
save paths (`copy_model`, `save_model`) do this. The trace `x` is stored as the amplitude ``a_j``; read the value
with `plasticity_value(syn, :LTPVars, :x, t)` (`monitor!(syn, [:x], :LTPVars)` does).

# Fields
- `A_LTD::FT = 8 * 10e-5pA / mV` (= 8e-4): LTD amplitude per mV.
- `A_LTP::FT = 14 * 10e-5pA / (mV * mV)` (= 1.4e-3): LTP amplitude per mV²; must be non-negative.
- `θ_LTD::FT = -70mV`: LTD threshold (also used for the filtered potential in the LTP term).
- `θ_LTP::FT = -49mV`: LTP threshold on the instantaneous potential.
- `τu::FT = 20ms`, `τv::FT = 7ms`, `τx::FT = 15ms`: time constants of ``u``, ``v``, ``x`` (`dt < τx` is required).
- `Wmax::FT = 30pF`, `Wmin::FT = 0.1pF`: weight bounds.
- `epoch::FT = 10`: restart interval of the accumulator in units of `τx` (`0 < epoch <= 25`).
- `active::VBT = [true]`: present in the parameter type; activation is controlled by the `active` field of
  `vSTDPVariables` (see `set_LTP!`).

# Precision
`FT = Float32` by default. The synapse converts the rule to the precision of its postsynaptic population
(`convert_precision`): a `Float64` population gives a `Float64` synapse, weights and state, whatever `FT` the rule was
built with; the values of a rule built with `Float32` fields are then those of the `Float32` numbers (8e-4 as
`Float32` is 8.0000002e-4 as `Float64`). Build the rule with `FT = Float64` and `Float64` values to have them exactly:
`vSTDPParameter(FT = Float64, A_LTD = 8e-4)`. The unit constants (`ms`, `pA`, `Hz`, ...) are `Float32`.

# Cost
Per step O(`Npost`) (filters, `g`, `K`); per presynaptic spike O(fan-out); once per epoch O(nnz).
No work per synapse per step. Serial.

# References
Clopath, C., Büsing, L., Vasilaki, E. & Gerstner, W. (2010). Connectivity reflects coding: a
model of voltage-based STDP with homeostasis. Nature Neuroscience 13, 344-352. The default
parameter values are not referenced in the code.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Poisson(N = 50, param = SNN.PoissonParameter(20Hz))
P = SNN.AdEx(N = 10, param = SNN.AdExParameter())
syn = SNN.SpikingSynapse(E, P, :ge; conn = (p = 0.2, μ = 1.0), LTPParam = SNN.vSTDPParameter())
SNN.train!(model = SNN.compose(; E, P, syn), duration = 200ms)
SNN.plasticity_value(syn, :W)      # current weights (syn.W is lazy)
```
"""
vSTDPParameter

@snn_kw struct vSTDPParameter{FT = Float32} <: LTPParameter
    A_LTD::FT = 8 * 10e-5pA / mV
    A_LTP::FT = 14 * 10e-5pA / (mV * mV)
    θ_LTD::FT = -70mV
    θ_LTP::FT = -49mV
    τu::FT = 20ms
    τv::FT = 7ms
    τx::FT = 15ms
    Wmax::FT = 30.0pF
    Wmin::FT = 0.1pF
    epoch::FT = 10.0f0
    active::VBT = [true]
end

"""
    vSTDPVariables(; Npre, Npost)

State of `vSTDPParameter`: `u`, `v` (filters of the postsynaptic potential, precision of the synapse, `Npost`),
`x` (STORED amplitudes ``a_j`` of the presynaptic traces, `Float64`, value ``x_j = a_j q^n``; read the
value with `plasticity_value`), `K` (accumulator per postsynaptic neuron), `snap` (snapshot of `K`
per synapse, allocated at the first step), the scratch `Wcur` of `plasticity_value(syn, :W)`, and the counters
and cached constants of the step (`steps`, `tick`, `matstep`, `pw`, `cache`). All `Float64` where
precision matters; see `vSTDPParameter`. Lists `:x` and `:W` in `lazy_variables`.
"""
vSTDPVariables

@snn_kw struct vSTDPVariables{VFT = Vector{Float32},VDT = Vector{Float64},VIT = Vector{Int},IT = Int} <: LTPVariables
    Npost::IT
    Npre::IT
    u::VFT = zeros(Npost)
    v::VFT = zeros(Npost)
    x::VDT = zeros(Npre)                  # STORED amplitude a_j (value = x * pw[1])
    K::VDT = zeros(Npost)                 # accumulator sum_m q^m g_i(m) of the epoch
    snap::VDT = Float64[]                 # per synapse: K at the previous presynaptic spike (allocated at first use)
    Wcur::VFT = Float32[]                 # scratch of plasticity_value(syn, :W)
    pw::VDT = [1.0, 1.0]                  # q^steps, q^-steps
    steps::VIT = [0]                      # steps since the epoch start
    tick::VIT = [0]                       # steps since the start (counted at the end of plasticity!)
    matstep::VIT = [-1]                   # tick at which the presynaptic spikes were materialised
    cache::VDT = fill(NaN, 9)             # dt, τx, A_LTP, Wmax, epoch, q, dt*A_LTP, epoch steps, 1/q
    active::VBT = [true]
    initialized::VBT = [false]
end

function plasticityvariables(param::T, Npre, Npost) where {T<:vSTDPParameter}
    param.A_LTP >= 0 || throw(ArgumentError("vSTDPParameter needs A_LTP >= 0 (the Wmax clamp commutes with the accumulation only for non-negative increments); use vSTDPDenseParameter for a negative A_LTP"))
    0 < param.epoch <= 25 || throw(ArgumentError("vSTDPParameter: epoch must be in (0, 25] (in units of τx; Float64 accumulator loses its digits after about 35 τx)"))
    param.τx > 0 || throw(ArgumentError("vSTDPParameter: τx must be positive"))
    return vSTDPVariables(Npre = Npre, Npost = Npost, VFT = _vft(param))
end

# the pending weight update and the trace amplitudes are stored, not values
lazy_variables(::vSTDPVariables) = (:x, :W)

function current_value(v::vSTDPVariables, param, key::Symbol, t::Real)
    key === :x && return convert(typeof(v.u), v.x .* v.pw[1])
    key === :W && throw(ArgumentError("the weights are not a field of vSTDPVariables: read them with plasticity_value(syn, :W)"))
    return getfield(v, key)
end

sync_weights!(c, v::vSTDPVariables) = (isnan(v.cache[1]) || _ev_rebase!(c, v); nothing)

# restart of the epoch: materialise every synapse, a_j -> x_j(now), K = 0, snapshots = 0
function _ev_rebase!(c, vars::vSTDPVariables)
    @unpack colptr, I, W = c
    @unpack x, K, snap, pw, steps, cache = vars
    T = eltype(W)
    dtA, Wmax, qp = cache[7], cache[4], pw[1]
    @inbounds for j in 1:length(colptr)-1
        a = dtA * x[j]
        @simd for s in colptr[j]:(colptr[j+1]-1)
            W[s] = T(min(Float64(W[s]) + a * (K[I[s]] - snap[s]), Wmax))
        end
        x[j] = x[j] * qp
    end
    fill!(K, 0.0)
    fill!(snap, 0.0)
    steps[1] = 0
    pw[1] = 1.0
    pw[2] = 1.0
    return nothing
end

# weights of the end of the last step: a scratch vector with the pending potentiation added and clamped
function current_weights(c, vars::vSTDPVariables)
    isnan(vars.cache[1]) && return c.W
    @unpack colptr, I, W = c
    @unpack x, K, snap, Wcur, cache = vars
    T = eltype(W)
    length(Wcur) == length(W) || resize!(Wcur, length(W))
    dtA, Wmax = cache[7], cache[4]
    @inbounds for j in 1:length(colptr)-1
        a = dtA * x[j]
        @simd for s in colptr[j]:(colptr[j+1]-1)
            Wcur[s] = T(min(Float64(W[s]) + a * (K[I[s]] - snap[s]), Wmax))
        end
    end
    return Wcur
end

# constants of the step; a change of dt, τx, A_LTP or Wmax (the factors of the accumulation) syncs first
function _ev_check!(c, vars::vSTDPVariables, param::vSTDPParameter, dt::AbstractFloat)
    ca = vars.cache
    if !(ca[1] == dt && ca[2] == param.τx && ca[3] == param.A_LTP && ca[4] == param.Wmax && ca[5] == param.epoch)
        isnan(ca[1]) || _ev_rebase!(c, vars)
        τx = Float64(param.τx)
        q = 1 - Float64(dt) / τx
        q > 0 || throw(ArgumentError("vSTDPParameter needs dt < τx"))
        ca[1] = dt; ca[2] = param.τx; ca[3] = param.A_LTP; ca[4] = param.Wmax; ca[5] = param.epoch
        ca[6] = q
        ca[7] = Float64(dt) * Float64(param.A_LTP)
        ca[8] = max(1.0, round(Float64(param.epoch) * τx / Float64(dt)))
        ca[9] = 1 / q
    end
    if length(vars.snap) != length(c.W)
        resize!(vars.snap, length(c.W)); fill!(vars.snap, 0.0)
        resize!(vars.Wcur, length(c.W))
        fill!(vars.K, 0.0)
        vars.steps[1] = 0; vars.pw .= 1.0
    end
    return nothing
end

# presynaptic spikes of the step: bring their outgoing weights to step n-1, take the snapshot of K,
# add the spike to the amplitude (x_j(n) includes it)
function _ev_spikes!(c, vars::vSTDPVariables, param::vSTDPParameter, spikes)
    @unpack colptr, I, W = c
    @unpack x, K, snap, pw, cache = vars
    T = eltype(W)
    dtA, Wmax, qi, itx = cache[7], cache[4], cache[9], 1 / Float64(param.τx)
    qn = pw[2] * qi                       # q^-n of the current step
    @inbounds for j in spikes
        a = dtA * x[j]
        @simd for s in colptr[j]:(colptr[j+1]-1)
            k = K[I[s]]
            W[s] = T(min(Float64(W[s]) + a * (k - snap[s]), Wmax))
            snap[s] = k
        end
        x[j] += qn * itx
    end
    vars.matstep[1] = vars.tick[1]
    return nothing
end

function update_traces!(
    c::PT,
    param::vSTDPParameter,
    vars::vSTDPVariables,
    dt::AbstractFloat,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    _ev_check!(c, vars, param, dt)
    _ev_spikes!(c, vars, param, presynaptic_spikes(c))
    return nothing
end

"""
    plasticity!(c::AbstractSparseSynapse, param::vSTDPParameter, variables::vSTDPVariables, dt::AbstractFloat, T::Time)

One step of `vSTDPParameter` (see its docstring): filters and accumulator of the postsynaptic
neurons, LTD at the presynaptic spikes, restart of the epoch when due. The weights of the spiking
neurons were materialised by `update_traces!` (done here if `update_traces!` was not called).
"""
function plasticity!(
    c::PT,
    param::vSTDPParameter,
    vars::vSTDPVariables,
    dt::AbstractFloat,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack colptr, I, W, v_post = c
    @unpack u, v, K, pw, steps, tick, matstep, cache, initialized = vars
    @unpack A_LTD, θ_LTD, θ_LTP, τu, τv, Wmin = param
    z = zero(eltype(W))
    if !initialized[1]
        u .= v_post
        v .= v_post
        initialized[1] = true
    end
    _ev_check!(c, vars, param, dt)
    spikes = presynaptic_spikes(c)
    matstep[1] == tick[1] || _ev_spikes!(c, vars, param, spikes)
    q, qi = cache[6], cache[9]
    pq = pw[1] * q
    pw[1] = pq
    pw[2] = pw[2] * qi
    @inbounds @fastmath for i in eachindex(u)
        u[i] += dt * (-u[i] + v_post[i]) / τu
        v[i] += dt * (-v[i] + v_post[i]) / τv
        g = max(v[i] - θ_LTD, z) * max(v_post[i] - θ_LTP, z)
        K[i] += pq * Float64(g)
    end
    @inbounds for j in spikes
        @simd for s in colptr[j]:(colptr[j+1]-1)
            w = W[s] - A_LTD * max(u[I[s]] - θ_LTD, z)
            W[s] = w < Wmin ? Wmin : w
        end
    end
    tick[1] += 1
    steps[1] += 1
    steps[1] >= cache[8] && _ev_rebase!(c, vars)
    return nothing
end

export vSTDPParameter, vSTDPVariables, plasticityvariables, plasticity!

@doc raw"""
    vSTDPDenseParameter(; A_LTD = 8e-4, A_LTP = 1.4e-3, θ_LTD = -70mV, θ_LTP = -49mV,
                          τu = 20ms, τv = 7ms, τx = 15ms, Wmax = 30pF, Wmin = 0.1pF, active = [true])

Clock-driven voltage-based STDP (Clopath et al. 2010), the dense reference of [`vSTDPParameter`](@ref) (it was called
`vSTDPParameter` up to SNNModels 1.9): the same rule, evaluated on every synapse at every step. Same equations,
update order, fields (without `epoch`) and precision rules as `vSTDPParameter`, but `syn.W` always holds the
current weights, and a negative `A_LTP` is allowed. The cost per step is O(nnz). Its state is `vSTDPDenseVariables`.
Use it as a reference, to check the event-driven rule, or when the weights must be read by code that does not go
through `plasticity_value`.

# Equations
With ``V_i`` the postsynaptic potential (`v_post`), ``u_i``, ``v_i`` its low-pass filters,
``x_j`` the presynaptic trace and ``[z]_+ = \max(z, 0)``:
```math
\begin{aligned}
τ_x \frac{dx_j}{dt} &= -x_j + \sum_k δ(t - t_j^k), \qquad
τ_u \frac{du_i}{dt} = -u_i + V_i, \qquad
τ_v \frac{dv_i}{dt} = -v_i + V_i,\\
Δw_{ij}^{LTD} &= -A_{LTD}\,[u_i - θ_{LTD}]_+ \quad \text{at each presynaptic spike of } j,\\
\frac{dw_{ij}^{LTP}}{dt} &= A_{LTP}\, x_j\, [v_i - θ_{LTD}]_+\, [V_i - θ_{LTP}]_+ .
\end{aligned}
```
A presynaptic spike increases ``x_j`` by ``1/τ_x`` (units 1/ms) and potentiation is a rate,
integrated over `dt`; the rule is independent of `dt`, as in Clopath et al. (2010). Note that
both thresholds of the LTP term are as in the code: the filtered potential ``v`` is compared
with `θ_LTD`, the instantaneous one with `θ_LTP`.

# Integration
`plasticity!` (only under `train!`), per step: (1) forward Euler for ``x`` (all presynaptic
neurons, including this step's spikes), (2) forward Euler for ``u`` and ``v`` (all
postsynaptic neurons, driven by the current `v_post`), (3) for each presynaptic neuron ``j``:
if ``j`` fired, LTD on its outgoing synapses and clamp at
`Wmin`; then, whether or not ``j`` fired, the LTP increment `dt * A_LTP * x * ...` on all its
outgoing synapses and clamp at `Wmax`. At the first call, the voltage traces ``u`` and ``v`` are
set to the current postsynaptic potential (they are not started at 0 mV).

!!! note "Changed after SNNModels 1.8.4"
    Up to 1.8.4 a spike increased ``x_j`` by ``dt/τ_x`` and the LTP increment was added per step
    without `dt`. The two factors cancel, so the weights are unchanged (same values up to
    rounding); only the scale of the recorded trace `x` changes (it was proportional to `dt`).
    The voltage traces started at 0 mV, which produced spurious depression during the first
    ``τ_u`` (about 0.02 of mean weight in 50 ms for a silent AdEx target with 20 Hz input).

# Fields
- `A_LTD::FT = 8 * 10e-5pA / mV` (= 8e-4): LTD amplitude per mV.
- `A_LTP::FT = 14 * 10e-5pA / (mV * mV)` (= 1.4e-3): LTP amplitude per mV².
- `θ_LTD::FT = -70mV`: LTD threshold (also used for the filtered potential in the LTP term).
- `θ_LTP::FT = -49mV`: LTP threshold on the instantaneous potential.
- `τu::FT = 20ms`: time constant of ``u`` (LTD filter).
- `τv::FT = 7ms`: time constant of ``v`` (LTP filter).
- `τx::FT = 15ms`: time constant of the presynaptic trace ``x``.
- `Wmax::FT = 30pF`, `Wmin::FT = 0.1pF`: weight bounds.
- `active::VBT = [true]`: present in the parameter type; activation is controlled by the
  `active` field of `vSTDPDenseVariables` (see `set_LTP!`).

# Precision
As for `vSTDPParameter`: the synapse converts the rule and its state to the precision of the postsynaptic
population. In `Float32` the per-step accumulation over 2e5 steps gives a relative error of 1e-5 against a `Float64`
reference; in `Float64` it agrees with the event-driven rule to about 1e-12.

# References
Clopath, C., Büsing, L., Vasilaki, E. & Gerstner, W. (2010). Connectivity reflects coding: a
model of voltage-based STDP with homeostasis. Nature Neuroscience 13, 344-352. The default
parameter values are not referenced in the code.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
E = SNN.Poisson(N = 50, param = SNN.PoissonParameter(20Hz))
P = SNN.AdEx(N = 10, param = SNN.AdExParameter())
syn = SNN.SpikingSynapse(E, P, :ge; conn = (p = 0.2, μ = 1.0), LTPParam = SNN.vSTDPDenseParameter())
SNN.train!(model = SNN.compose(; E, P, syn), duration = 200ms)
```
"""
vSTDPDenseParameter

@snn_kw struct vSTDPDenseParameter{FT = Float32} <: LTPParameter
    A_LTD::FT = 8 * 10e-5pA / mV
    A_LTP::FT = 14 * 10e-5pA / (mV * mV)
    θ_LTD::FT = -70mV
    θ_LTP::FT = -49mV
    τu::FT = 20ms
    τv::FT = 7ms
    τx::FT = 15ms
    Wmax::FT = 30.0pF
    Wmin::FT = 0.1pF
    active::VBT = [true]
end

"""
    vSTDPDenseVariables(; Npre, Npost, u = zeros(Npost), v = zeros(Npost), x = zeros(Npre), active = [true],
                          initialized = [false])

State of `vSTDPDenseParameter`: `u`, `v` (low-pass filters of the postsynaptic potential with
time constants `τu`, `τv`, length `Npost`, mV), `x` (presynaptic spike trace, length `Npre`,
1/ms), the `active` flag and `initialized`: while `initialized[1]` is false, the first
`plasticity!` call sets `u` and `v` to the postsynaptic potential.
"""
vSTDPDenseVariables

@snn_kw struct vSTDPDenseVariables{VFT = Vector{Float32},IT = Int} <: LTPVariables
    ## Plasticity variables
    Npost::IT
    Npre::IT
    u::VFT = zeros(Npost) # postsynaptic potential filtered with τu (LTD)
    v::VFT = zeros(Npost) # postsynaptic potential filtered with τv (LTP)
    x::VFT = zeros(Npre) # presynaptic spike trace
    active::VBT = [true]
    initialized::VBT = [false] # u, v are set to v_post at the first plasticity! call
end

function plasticityvariables(param::T, Npre, Npost) where {T<:vSTDPDenseParameter}
    return vSTDPDenseVariables(Npre = Npre, Npost = Npost, VFT = _vft(param))
end

"""
    plasticity!(c::AbstractSparseSynapse, param::vSTDPDenseParameter, variables::vSTDPDenseVariables, dt::AbstractFloat, T::Time)

One step of the dense voltage-based STDP rule `vSTDPDenseParameter` (see its docstring for the
equations and the update order): updates the traces in `variables` and the weights `c.W` in
place, clamping them to `[Wmin, Wmax]`. Called by `train!`, never by `sim!`.
"""
function plasticity!(
    c::PT,
    param::vSTDPDenseParameter,
    plasticity::vSTDPDenseVariables,
    dt::AbstractFloat,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, v_post, fireJ, g, index = c
    @unpack u, v, x, initialized = plasticity
    @unpack A_LTD, A_LTP, θ_LTD, θ_LTP, τu, τv, τx, Wmax, Wmin = param
    z = zero(eltype(W))
    if !initialized[1]
        u .= v_post
        v .= v_post
        initialized[1] = true
    end
    dtA_LTP = dt * A_LTP

    # update pre-synaptic spike trace
    @fastmath @inbounds begin
        for j in eachindex(fireJ) # Iterate over all columns, j: presynaptic neuron
            x[j] += -dt * x[j] / τx + fireJ[j] / τx
        end

        Is = 1:(length(rowptr)-1)
        @turbo for i in eachindex(Is) # Iterate over postsynaptic neurons
            u[i] += dt * (-u[i] + v_post[i]) / τu # postsynaptic neuron
            v[i] += dt * (-v[i] + v_post[i]) / τv # postsynaptic neuron
        end
        for j in eachindex(fireJ) # Iterate over presynaptic neurons
            if fireJ[j]
                @turbo for s = colptr[j]:(colptr[j+1]-1)
                    W[s] += - A_LTD * clamp(u[I[s]] - θ_LTD, z, Inf)
                    W[s] < Wmin && (W[s] = Wmin)
                end
            end
            @turbo for s = colptr[j]:(colptr[j+1]-1)
                W[s] += dtA_LTP *
                    x[j] *
                    clamp(v[I[s]] - θ_LTD, z, Inf) *
                    clamp(v_post[I[s]] - θ_LTP, z, Inf)
                W[s] > Wmax && (W[s] = Wmax)
            end
        end

    end
end

export vSTDPDenseParameter, vSTDPDenseVariables
