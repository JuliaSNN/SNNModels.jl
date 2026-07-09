# SNNModels.jl — Library Map

> Purpose: run biophysical SNN simulations and analyse them.
> Generated 2026-06-06. All core source files read directly.

---

## 1. Module layout

```
SNNModels.jl/src/
├── utils/
│   ├── macros.jl       @snn_kw, @update!, @update, @symdict, update_with_merge
│   ├── unit.jl         Float32 unit scale factors (ms, mV, nS, pA, pF, Hz, um, mm…)
│   ├── structs.jl      core abstract types, Time, Spiketimes, VBT/VIT aliases, validation
│   ├── main.jl         sim! / train! outer loops + innermost dispatch
│   ├── util.jl         compose, extract_items, print_model, remove_element
│   ├── copying.jl      modelcopy (deep copy with empty record interception)
│   ├── perturbation.jl perturbation_test, perturbation_record, clear_perturbation_records!, clear_perturbation_monitor!
│   ├── graph.jl        graph(model) → MetaDiGraph, find_id_vertex, filter_edge_props
│   ├── record.jl       monitor!, record, firing_rate (dispatch), getvariable, clear_records!
│   ├── io.jl           SNNsave / SNNload / SNNfolder / SNNpath (DrWatson convention)
│   ├── sparse_matrix.jl  connect!, matrix, update_weights!, presynaptic, postsynaptic
│   └── spatial.jl      place_populations, periodic_distance, compute_connections,
│                        gaussian_weight, spatial_activity, linear_network
├── populations/
│   ├── populations.jl  abstract types, default integrate!/plasticity!, Population dispatcher
│   ├── poisson.jl      Poisson (fixed rate)
│   ├── inhomogeneous_poisson.jl  InhomogeneousPoisson (OU noise-modulated)
│   ├── identity.jl     Identity (g→fire passthrough)
│   ├── iz.jl           Izhikevich
│   ├── hh.jl           Hodgkin-Huxley
│   ├── morrislecar.jl  MorrisLecar
│   ├── rate.jl         rate-coded population
│   ├── wilsoncowan.jl  WilsonCowan mean-field
│   ├── hetrec.jl       HetRec (heterogeneous dendritic timescales)
│   ├── spike/postspike.jl  PostSpike (threshold adaptation)
│   ├── synapse/        Receptor, Receptors, ReceptorSynapse, DoubleExpSynapse,
│   │                   SingleExpSynapse, DeltaSynapse, CurrentSynapse,
│   │                   DoubleExpCurrentSynapse, Confraveux2025
│   ├── generalized_if/
│   │   ├── if.jl       IF (leaky I&F, also serves as AdEx with a=b=0)
│   │   ├── adex.jl     AdEx (adaptive exponential)
│   │   ├── gif.jl      GIF (generalized I&F)
│   │   └── if_extended.jl  IFExtended
│   └── multicompartment/
│       ├── dendrite.jl         Dendrite struct, Physiology, G_axial/G_mem/C_mem
│       ├── dendneuron_parameter.jl  DendNeuronParameter
│       ├── tripod.jl           Tripod (soma + 2 dendrites)
│       └── ballandstick.jl     BallAndStick (soma + 1 dendrite)
├── connections/
│   ├── connections.jl       abstract types, default forward!/plasticity!/update_traces!
│   ├── spiking_synapse.jl   SpikingSynapse (core), SpikingSynapseDelayParameter
│   ├── sparse_plasticity.jl LTP/STP dispatch, NoLTP/NoSTP, set_LTP!/set_STP!
│   ├── sparse_plasticity/
│   │   ├── STP.jl           MarkramSTPParameter (Event/Timestep/Het)
│   │   ├── STDP_traces.jl   STDPGerstner, STDPConfavreux2025, STDPMexicanHat
│   │   ├── iSTDP.jl         iSTDPRate, iSTDPTime, iSTDPPotential
│   │   ├── vSTDP.jl         vSTDPParameter
│   │   ├── CaRule.jl        CaPlasticityParameter (legacy; uses c.plasticity)
│   │   ├── STDP_structured.jl  STDPSymmetric, STDPAntiSymmetric (Festa 2024)
│   │   └── dump.jl          (utility/debug)
│   ├── metaplasticity/
│   │   ├── normalization.jl    SynapseNormalization (MultiplicativeNorm/AdditiveNorm)
│   │   ├── aggregate_scaling.jl  AggregateScaling
│   │   └── turnover.jl       Turnover (RandomTurnover/ActivityDependentTurnover)
│   ├── fl_synapse.jl        FLSynapse (FORCE learning, rate-coded)
│   ├── fl_sparse_synapse.jl FLSparseSynapse
│   ├── rate_synapse.jl      RateSynapse (rate→rate)
│   ├── pinning_synapse.jl   PinningSynapse
│   ├── pinning_sparse_synapse.jl  PinningSparseSynapse
│   ├── spike_rate_synapse.jl  SpikeRateSynapse
│   └── empty.jl             EmptySynapse, NoLTP, NoSTP, NoVariables
├── stimuli/
│   ├── stimuli.jl       abstract types, set_variable!/set_intervals!/set_active!, neurons()
│   ├── poisson.jl       PoissonFixed, PoissonVariable, PoissonInterval → PoissonStimulus
│   ├── poisson_layer.jl PoissonLayer, PoissonLayerHet → PoissonStimulusLayer
│   ├── current.jl       CurrentStimulus
│   ├── timed.jl         SpikeTimeParameter, SpikeTimeStimulusParameter → SpikeTimeStimulus
│   ├── balanced.jl      BalancedParameter → BalancedStimulus
│   ├── stimulus_group.jl  StimulusGroup
│   ├── variable_inputs.jl VariableInputParameter (func-driven current; INCOMPLETE)
│   └── empty.jl         EmptyStimulus
└── analysis/
    ├── spikes.jl        spiketimes, firing_rate, bin_spiketimes, gaussian_smooth, kernels,
    │                    spikes_in_interval, ISI_CV2, interval_standard_spikes!
    ├── targets.jl       STTC (pair + matrix), tile_interval, _tile_fraction,
    │                    _coincident_fraction, _sttc_pair, is_unimodal
    └── populations.jl   population_indices, filter_items, subpopulations,
                         target_neurons, average_conn_strength
```

---

## 2. Core types (`utils/structs.jl`)

```julia
abstract type AbstractParameter end
abstract type AbstractComponent end
abstract type AbstractPopulation <: AbstractComponent end
abstract type AbstractConnection <: AbstractComponent end
abstract type AbstractStimulus <: AbstractComponent end
abstract type AbstractStimulusGroup <: AbstractGroup end
Component = Union{AbstractPopulation, AbstractConnection, AbstractStimulus}
NetworkModel = NamedTuple  # alias; compose returns this

Spiketimes = Vector{Vector{Float32}}  # one inner vec per neuron
VBT = Vector{Bool}
VIT = Vector{Int}

mutable struct Time
    t::Vector{Float32}   # [current_time_ms]; vector for mutability
    tt::Vector{Int32}    # [current_timestep]
    dt::Float32          # default 0.125ms
end
```

Required fields for every component: `id::String`, `name::String`, `param`, `records::Dict`.
`isa_model(model)` validates the NamedTuple has `pop`, `syn`, `stim`, `time`, `name`.

---

## 3. The simulation loop (`utils/main.jl`)

### Two entry points

| Function | Plasticity/traces? | Use when |
|---|---|---|
| `sim!(model, duration)` | No | read-out / analysis runs |
| `train!(model, duration)` | Yes | learning, STDP, STP burn-in |

### Per-dt dispatch

```
train! inner step                      sim! inner step
─────────────────────────────────────  ──────────────────────────────────
record_zero!(P, C, S, T)               record_zero!(P, C, S, T)
update_time!(T, dt)                    update_time!(T, dt)

for s in S                             for s in S
  stimulate!(s, s.param, T, dt)          stimulate!(s, s.param, T, dt)
  record!(s, T)                          record!(s, T)

for p in P                             for p in P
  update_traces!(p, p.param, dt, T)      integrate!(p, p.param, dt)
  integrate!(p, p.param, dt)             record!(p, T)
  plasticity!(p, p.param, dt, T)
  record!(p, T)

for c in C                             for c in C
  update_traces!(c, c.param, dt, T)      forward!(c, c.param, dt, T)
  forward!(c, c.param, dt, T)            record!(c, T)
  plasticity!(c, c.param, dt, T)
  record!(c, T)
```

Order: **stimuli → populations → connections** every dt.
`record_zero!` only fires at t=0 (seeds the record buffer).

### Calling conventions

```julia
model = compose(; exc, pv, sst, exc_pv, pv_exc, stim_th)
monitor!(model.pop.exc, [:v_s, :fire, :w_s]; sr = 1kHz)
sim!(model, 2s; dt = 0.125ms, pbar = true)
train!(model, 10s)
get_time(model)        # → simulated time in ms
reset_time!(model)     # → 0 (for warmup → recording transitions)
initialize!(; model)   # alias for train!
```

---

## 4. Populations

### Abstract interface

```julia
abstract type AbstractPopulationParameter <: AbstractParameter end

integrate!(p, param, dt)            # mandatory: update neuron state
plasticity!(p, param, dt, T)        # optional: intrinsic plasticity (default: nothing)
update_traces!(p, param, dt, T)     # optional: trace update (default: nothing)
Population(param; kwargs...)        # constructor dispatch on param type
```

Required fields: `id, name, N, param, records`.

### Population types

| Type | File | Key state vars | Key params |
|---|---|---|---|
| `Identity` | identity.jl | g, h, fire | — |
| `Poisson` | poisson.jl | fire, r | `rate` |
| `InhomogeneousPoisson` | inhomogeneous_poisson.jl | fire, r, noise | `β, τ, r0, rate_timescale` |
| `IZ` (Izhikevich) | iz.jl | v, u, fire | `a, b, c, d` |
| `HH` (Hodgkin-Huxley) | hh.jl | v, m, h, n, fire | standard HH |
| `MorrisLecar` | morrislecar.jl | V, W, fire | `gCa, gK, gL` |
| `RatePopulation` | rate.jl | r, fire | `τ, gain` |
| `WilsonCowan` | wilsoncowan.jl | E, I | `τE, τI, c++` |
| `IF` | generalized_if/if.jl | v, w, fire | `C, gl, τm, Vt, Vr, El, a, b, τw` |
| `AdEx` | generalized_if/adex.jl | v, w, fire, θ | `C, gl, Vt, Vr, El, ΔT, τw, a, b` |
| `GIF` | generalized_if/gif.jl | — | — |
| `IFExtended` | generalized_if/if_extended.jl | — | — |
| `Tripod` | multicompartment/tripod.jl | v_s, w_s, v_d1, v_d2 | DendNeuronParameter + AdExParameter |
| `BallAndStick` | multicompartment/ballandstick.jl | v_s, w_s, v_d | DendNeuronParameter + AdExParameter |
| `HetRec` | hetrec.jl | v, r, dendrites | HetRecParameter |

### IF and AdEx (most used)

```julia
# IF: C=-1, gl=-1 (sentinel); τm = C>0&&gl>0 ? C/gl : 15ms
# → call with explicit C,gl or leave τm at default 15ms
IFParameter: C, gl, τm, Vt, Vr, El, ΔT, a, b, τw
IF state:    v, w, fire, θ, tabs, I, syn_curr, synvars, receptors

# AdEx: Brette-Gerstner 2005 defaults
AdExParameter: C=281pF, gl=40nS, Vt=-50mV, Vr=-70.6mV, El=-70.6mV, ΔT=2mV, τw=144ms, a=4nS, b=80.5pA
AdEx state:    v, w, fire, θ, tabs, I, syn_curr, synvars, receptors
```

Integration (both):
```
v += dt/τm * (-(v-El) + ΔT·exp((v-θ)/ΔT) - R·syn_curr - R·w + R·I)
w += dt/τw * (a·(v-El) - w)
# spike: v→20mV, v→Vr; w += b; θ += At; tabs set
```
Two dispatch methods per model: scalar params `FT=Float32` and heterogeneous params `FT=Vector{Float32}`.

### Tripod (main biophysical model)

```julia
DendNeuronParameter(ds = [(l1, d1), (l2, d2)])  # dendritic lengths + diameters in um
# → computes Dendrite structs: El, C, gax, gm, l, d
# → uses Physiology constants (human_dend or mouse_dend): Ri, Rd, Cd

Tripod fields:
  v_s, w_s              # soma
  v_d1, v_d2            # two dendrites (Heun integration)
  synvars_s/d1/d2       # conductance state per compartment
  receptors_s/d1/d2     # (glu, gaba) named tuple per compartment

# Synaptic targeting by compartment:
SpikingSynapse(pre, post, :glu, :d1; ...)   # → receptors_d1.glu
SpikingSynapse(pre, post, :gaba, :s;  ...)   # → receptors_s.gaba
```

`BallAndStick` is identical but with 1 dendrite (`v_d`, `synvars_d`, `receptors_d`).

### HetRec

Non-recurrent population with heterogeneous dendritic timescales. Designed for temporal feature learning in optimization experiments.

```julia
HetRecParameter:
  Nd        # dendritic compartments per neuron
  overlap   # 0=private dendrites, 1=fully shared across neurons
  τd        # Distribution for dendritic τ (e.g. Uniform(10ms, 100ms))
  rate      # Distribution for baseline firing rate
  steepness # nonlinearity slope
  τm, τabs, τrate
```

### PostSpike

Spike-triggered threshold adaptation, attached to every spiking population.

```julia
PostSpike: At=0mV (threshold jump), τA=10ms (decay), AP_membrane=10mV, τabs=1ms
# fire: θ += At; tabs = τabs/dt; v → 20mV → Vr
```

### Dendrite biophysics (`dendrite.jl`)

```julia
Physiology(Ri, Rd, Cd)        # cylinder physical constants
human_dend = Physiology(200Ω·cm, 38907Ω·cm², 0.5μF/cm²)
mouse_dend = Physiology(200Ω·cm, 1700Ω·cm², 1μF/cm²)

G_axial(; Ri, d, l)  # axial conductance (nS) = π·d²/(4·Ri·l)
G_mem(;   Rd, d, l)  # membrane conductance  = π·d·l/Rd
C_mem(;   Cd, d, l)  # capacitance (pF)      = Cd·π·d·l

Dendrite struct: El, C, gax, gm, l, d  (all as Vector{Float32} over N neurons)
```

### Synaptic models (`populations/synapse/`)

| Type | Variables | Use |
|---|---|---|
| `DoubleExpSynapse` | ge, gi, he, hi | IF/AdEx default; τre/τde, τri/τdi, E_e/E_i |
| `SingleExpSynapse` | ge, gi | simpler, single τ |
| `ReceptorSynapse` | glu, gaba per compartment | Tripod/BallAndStick dendrites |
| `CurrentSynapse` | current injection | no conductance |
| `DeltaSynapse` | instantaneous | no decay |
| `DoubleExpCurrentSynapse` | current with double-exp | |
| `Confraveux2025` | — | specific plasticity coupling |

`Receptors(AMPA, NMDA, GABAa, GABAb)` → `ReceptorArray = Vector{Receptor}`.
`Receptor(E_rev, τr, τd, g0)` — precomputes `gsyn, α, τr⁻, τd⁻`.
`ReceptorVoltage = Receptor` — `nmda=1.0f0` activates voltage-dependent Mg²⁺ block.
`NMDAVoltageDependency` — parameters for Mg²⁺ block curve.

`synaptic_variables(synapse, N)` → allocates conductance state vectors.
`synaptic_receptors(synapse, N)` → NamedTuple `(glu, gaba)` of g arrays, pointed at by SpikingSynapse.

---

## 5. Synapses (Connections)

### Abstract interface

```julia
abstract type AbstractSpikingSynapse <: AbstractSparseSynapse end
abstract type PlasticityParameter end
abstract type PlasticityVariables end

forward!(c, param, dt, T)        # mandatory: spikes → conductances
plasticity!(c, param, dt, T)     # optional (default: nothing)
update_traces!(c, param, dt, T)  # optional (default: nothing)
```

### SpikingSynapse — the core synapse

```julia
fields:
  rowptr, colptr, I, J, index  # CSR sparse representation (Npost×Npre)
  W::VFT                        # weights (modified in-place by LTP)
  ρ::VFT                        # STP modulation per pre-neuron (1.0 without STP)
  fireI::VBT                    # ref to post.fire
  fireJ::VBT                    # ref to pre.fire
  g::VFT                        # ref into post.receptors_*.glu or .gaba
  v_post::VFT                   # ref to post.v or post.v_s (used by vSTDP)
  LTPParam, LTPVars             # long-term plasticity
  STPParam, STPVars             # short-term plasticity
  targets::Dict                 # :fire→pre.id, :post→post.id, :sym, :type

constructor:
  SpikingSynapse(pre, post, sym, [comp];
                 conn = (p=0.1f0, μ=1nS),     # OR conn=(w=W_matrix,)
                 LTPParam = NoLTP(),
                 STPParam = NoSTP(),
                 delay_dist = nothing,
                 name = "SpikingSynapse")
```

`forward!` hot path (no delay):
```julia
for j in eachindex(fireJ)
    if fireJ[j]
        for s in colptr[j]:(colptr[j+1]-1)
            g[I[s]] += W[s] * ρ[s]   # CSR column scan
        end
    end
end
```
With delay: spikes buffered per-post, delivered when `t ≥ spike_time`.

### Connectivity specification

```julia
conn = (p=0.1f0, μ=1nS)              # random Bernoulli, mean weight
conn = (p=0.1f0, μ=1nS, σ=0.1nS)    # with std dev
conn = (w = my_matrix,)              # explicit dense matrix
conn = LinearAlgebra.I(N)            # 1-to-1 identity
conn = diagm(fill(w, N))             # diagonal
```

`sparse_matrix(Npre, Npost, conn)` → SparseMatrix → `dsparse(w)` → CSR arrays.

### Plasticity dispatch (`sparse_plasticity.jl`)

```julia
# Both STP and LTP are gated by their active flag:
any(c.STPVars.active) && plasticity!(c, c.STPParam, c.STPVars, dt, T)
any(c.LTPVars.active) && plasticity!(c, c.LTPParam, c.LTPVars, dt, T)

# Runtime on/off:
set_LTP!(syn, true/false)
set_STP!(syn, true/false)
set_plasticity!(syn, param, state)
```

### Long-term plasticity (LTP)

| Type | Trace update | Weight update | Threading | Key params |
|---|---|---|---|---|
| `NoLTP` | — | — | — | — |
| `STDPGerstner` | **event-driven** `exp(-(t-last)/τ)` | post-pre + pre-post | `Threads.@threads` chunks | `A_pre, A_post, τpre, τpost, Wmax, Wmin` |
| `STDPConfavreux2025` | **event-driven** same | post-pre + pre-post × (η, α, β, κ, γ) | `Threads.@threads` chunks | `η, α, β, κ, γ, τpre, τpost, Wmax, Wmin` |
| `STDPMexicanHat` | **continuous** dt decay `@turbo` | row+col scan on spike | none | `A, τ, Wmax, Wmin` |
| `vSTDPParameter` | **continuous** u(post), v(post), x(pre) `@turbo` | `Threads.@threads` over pre chunks | yes | `A_LTD, A_LTP, θ_LTD, θ_LTP, τu, τv, τx, Wmax, Wmin` |
| `iSTDPRate` | **continuous** dt decay `@turbo` | clamp on spike `@turbo` col/row | none | `η, r, τy, Wmax, Wmin` |
| `iSTDPPotential` | **continuous** tpost tracks v_post | on spike, no `@turbo` | none | `η, v0, τy, Wmax, Wmin` |
| `iSTDPTime` | **continuous** same as Rate | — | none | `η, τy, Wmax, Wmin` |
| `CaPlasticityParameter` | **continuous** dt decay `@turbo` | row+col scan on spike | none | `A_pre, A_post, τpre, τpost, Wmax, Wmin` |
| `STDPSymmetric` | **continuous** to_x/tr_x `@turbo` | row+col on spike | none | `A_x, A_y, τ_x, τ_y, αpre, αpost, Wmax, Wmin` |
| `STDPAntiSymmetric` | same | same | none | same |

Variables structs:
- `STDPVariables(Npre, Npost)`: `tpre, tpost, last_pre, last_post, Δpre, Δpost`
- `iSTDPVariables(Npre, Npost)`: `tpre, tpost, last_spike`
- `vSTDPVariables(Npre, Npost)`: `u, v, x` (all Npost-sized ← **x should be Npre**; see §10.3)
- `STDPStructuredVariables(Npre, Npost)`: `to_x, to_y` (Npost), `tr_x, tr_y` (Npre)

### Short-term plasticity (STP)

Modulates `ρ` (per pre-neuron). `forward!` uses `W[s] * ρ[s]`.

| Type | Update | Key params |
|---|---|---|
| `NoSTP` | nothing, ρ=1 | — |
| `MarkramSTPParameterEvent` (= `MarkramSTPParameter`) | event-driven: last_spike, u, x | `τD, τF, U` |
| `MarkramSTPParameterTimestep` | timestep dt-based | same |
| `MarkramSTPParameterHet` | event-driven, per-synapse | `τD::VFT, τF::VFT, U::VFT` |

Variables: `MarkramSTPVariables`: u (utilization), x (resources), _ρ=u·x, last_spike.
At spike j: `u[j] = u[j]·exp(-Δt/τF) + U·(1-u[j]·exp(-Δt/τF))`, `x[j] -= u[j]·x[j]`, `ρ[j] = u[j]·x[j]`.

### Metaplasticity (added to model.syn alongside synapses)

| Type | Purpose | Key params |
|---|---|---|
| `SynapseNormalization` | Renormalize W per post-neuron | `MultiplicativeNorm(τ)` or `AdditiveNorm(τ)` |
| `AggregateScaling` | Scale W to maintain target firing rate | `τ, τa, τe, Y (target rates), Wmin, Wmax` |
| `Turnover` | Prune/grow synapses | `RandomTurnover(rate, threshold)` or `ActivityDependentTurnover(rate, fraction, τpre, τpost)` |

All are `<: AbstractMetaPlasticity <: AbstractConnection`. The graph builder links them to their target synapses via `targets[:synapses]`.

### Other connection types

| Type | File | Use |
|---|---|---|
| `FLSynapse` | fl_synapse.jl | FORCE learning; rate-coded; dense W, P(inverse corr), u, w, z |
| `FLSparseSynapse` | fl_sparse_synapse.jl | sparse FORCE variant |
| `RateSynapse` | rate_synapse.jl | rate → rate coupling |
| `PinningSynapse` | pinning_synapse.jl | pinned (frozen) weights |
| `PinningSparseSynapse` | pinning_sparse_synapse.jl | sparse pinning |
| `SpikeRateSynapse` | spike_rate_synapse.jl | spike pre → rate post |
| `EmptySynapse` | empty.jl | no-op placeholder |

`FLSynapse.forward!` computes `g += W·rJ` and optionally the FORCE recursive least-squares update on P, u, w.

---

## 6. Stimuli

### Abstract interface

```julia
stimulate!(s, s.param, T, dt)   # called every step before populations

# Runtime control (work on any AbstractStimulus with matching param fields):
set_variable!(stim, :rate, new_rate)
set_intervals!(stim, intervals)
set_active!(stim, true/false)
neurons(stim)                   # → vector of targeted neuron indices
```

### Stimulus types

| Stimulus struct | Param type | Drive |
|---|---|---|
| `PoissonStimulus` | `PoissonFixed(rate, μ)` | fixed-rate Poisson → g |
| `PoissonStimulus` | `PoissonVariable(variables, rate::Function, μ)` | function-driven rate |
| `PoissonStimulus` | `PoissonInterval(rate, intervals)` | active only within intervals |
| `PoissonStimulusLayer` | `PoissonLayer(rate; N)` or `PoissonLayerHet(rates)` | Poisson with sparse connectivity |
| `CurrentStimulus` | — | I injection |
| `SpikeTimeStimulus` | `SpikeTimeStimulusParameter(spiketimes, neurons)` | pre-recorded spike trains |
| `BalancedStimulus` | `BalancedParameter(kIE, β, τ, r0, w, wIE, same_input)` | balanced E/I via OU noise |
| `StimulusGroup` | — | container of AbstractStimulus elements |
| `EmptyStimulus` | — | no-op |

**`SpikeTimeStimulus`** (timed.jl):
- CSR sparse structure identical to SpikingSynapse
- Delivers to `g` at exact times via `next_spike`/`next_index` cursors (O(1) per step, no search)
- `SpikeTimeParameter` constructors:
  ```julia
  SpikeTimeParameter(times::Vector{Float32}, neurons::Vector{Int})   # flat sorted
  SpikeTimeParameter(spiketimes::Spiketimes)                         # Vector{Vector{Float32}}
  SpikeTimeParameter(spiketimes::Vector{Vector{Float64}})            # auto-converts
  ```

**`BalancedStimulus`** (balanced.jl):
- Maintains separate `ge, gi` conductance vectors
- `kIE`: inhibitory scaling factor; `β`: OU noise amplitude; `same_input`: broadcast vs per-neuron

**`VariableInputParameter`** (variable_inputs.jl):
- `VariableInputParameter(variables::Dict, target::Symbol, func::Function)`
- `stimulate!` calls `func(variables, t, neuron_id)` → I[i] each step
- Helper functions: `ramping_current`, `OrnsteinUhlenbeckProcess`, `SinWaveNoise`
- **INCOMPLETE**: `stimulate!` dispatches on `VariablesParameter` which is undefined in this file

---

## 7. Utilities

### Macros (`utils/macros.jl`)

| Macro | What it does |
|---|---|
| `@snn_kw struct T{FT=Float32}` | **Custom** keyword constructor. Infers type params (FT) from field defaults. Not `@with_kw`. |
| `@update! base begin x.y.z = val end` | Deep in-place update of nested NamedTuple/struct in caller scope. Returns modified base. |
| `@update base ...` | Immutable version; returns new config without modifying base. |
| `@symdict(a, b, c)` | `Dict(:a=>a, :b=>b, :c=>c)` |

`update_with_merge(nt, path::Vector{Symbol}, value)`:
- Recurses into NamedTuple; calls `merge(sub, (key => updated,))` at each level
- Also handles plain structs by converting to NT → updating → `typename.wrapper(; nt...)`
- Warns (not errors) if field missing

### Units (`utils/unit.jl`)

All Float32 scale factors used directly in param literals:
`ms, s, mV, nS, pA, pF, Hz, kHz, um, mm, Ω, cm, μF`

### Model construction (`utils/util.jl`)

```julia
compose(; exc, pv, sst, exc_pv, pv_exc, stim_th; silent=false, name=..., time=Time())
# classifies each kwarg by supertype: →pop / →syn / →stim
# returns (pop=(…), syn=(…), stim=(…), name, time)

remove_element(model, key)       # returns new model without that component
remove_element(model, keys::Vector)
merge_models(…)                  # deprecated; use compose
```

### Sparse matrix utilities (`utils/sparse_matrix.jl`)

```julia
connect!(c, j, i, μ)            # add/modify weight for pre j → post i
matrix(c)                        # reconstruct SparseMatrix from CSR
matrix(c, sym)                   # for any field (ρ, etc.)
matrix(c, sym, time)             # time-sliced from recording → SparseMatrix
matrix(c, sym, times::Vector)    # → 3D array
update_weights!(c, j, i, w)      # update single weight by (j,i) index
update_weights!(c, js, is, w)    # batch update
presynaptic(c)                   # → Vector{Vector} of pre indices for each post
presynaptic(c, i)                # pre indices for post neuron i
postsynaptic(c)                  # → Vector{Vector} of post indices for each pre
postsynaptic(c, j)               # post indices for pre neuron j
```

### Spatial utilities (`utils/spatial.jl`)

```julia
place_populations(Npop, grid_size)
# → NamedTuple of random 2D positions per population

periodic_distance(p1, p2, grid_size)
# toroidal Euclidean distance (wraps at grid boundaries)

neurons_within_circle(points, center, distance, grid_size)
# boolean mask of neurons within radius

gaussian_weight(pre, post; σx, σy, grid_size)
# normalized Gaussian connectivity weight

compute_connections(pre, post, points; conn, spatial, dist)
# builds sparse (L::BitMatrix, W::Matrix, P::Matrix) with two spatial modes:
#   :critical_distance  → short-range p_short + long-range p_long
#   :gaussian           → Gaussian probability profile, normalized by γ
# weight sampled from dist::Sampleable (e.g. Normal, Uniform)

linear_network(N; σ_w, w_max)
# ring attractor weight matrix: Gaussian on circular topology

spatial_activity(points, activity; T, L or N, grid_size)
# 3D array (x_tiles, y_tiles, time_groups) of spatially-averaged activity
# T: scalar (group size) or Vector (explicit time index groups)
# L: tile size (physical), N: number of tiles
# Returns: spatial_avg, x_range, y_range
```

### Recording (`utils/record.jl`)

```julia
monitor!(p, syms; sr=1kHz, variables=nothing)
# enables recording for listed fields at sample rate sr
# variables= adds a prefix to the key (e.g. :synvars_d1)

record(p, :v_s; interval=0s:1ms:2s, interpolate=true, range=false)
# → ScaledInterpolation (callable as v(neurons, interval))
record(p, :fire; interval=...)           # → firing_rate internally
record(p, :spiketimes)                   # → spiketimes(p)
record(pops::Vector, sym; interval)      # concatenates across pops (vcat)
getvariable(p, sym)                      # raw array (no interpolation)
clear_records!(model)                    # free memory

interpolated_record(p, sym)              # returns ScaledInterpolation + interval range
get_interpolator(A)                      # BSpline(Linear()) per dim, NoInterp for singletons
```

`fire` recording format: `p.records[:fire][:time]` (timestep index vector), `[:neurons]` (vector of vectors).
All other fields: `p.records[sym]` → vector of snapshots pushed each recorded step.

### IO (`utils/io.jl`)

DrWatson-based naming convention:
```julia
SNNfolder(path, name, info)   # path/savename(name, info, "-")
SNNfile(type, count, suffix)  # "type-N-suffix.jld2"
SNNpath(path, name, info, type, count)  # full path
SNNload(; path, name, info, count, type=:model)
SNNsave(model, path, ...)
```

---

## 8. Analysis

### Spikes (`analysis/spikes.jl`)

```julia
spiketimes(p; interval)
# → Spiketimes from p.records[:fire]; optionally filtered to interval

firing_rate(spiketimes; interval, kernel=alpha_kernel, interpolate=true,
            pop_average=false, time_average=false, neurons=:ALL)
# convolves spike trains with kernel; returns ScaledInterpolation or matrix
# time_average=true → scalar or vector of mean rates (fast path, no convolution)

firing_rate(p, interval; ...)       # shortcut: spiketimes(p) then above
firing_rate(pops::Vector, interval; ...)  # concatenates

bin_spiketimes(spikes; interval)    # → spike_train array + interval

gaussian_smooth(xs, x::Vector, σ)   # boundary-renormalized Gaussian kernel
                                    # truncated at 3σ, @inbounds @fastmath

alpha_kernel(; interval, kwargs...)  # alpha-function convolution kernel
exponential_kernel(; ...)            # exponential kernel

spikes_in_interval(spiketimes, interval)   # filter
ISI_CV2(spiketimes)                        # regularity metric
interval_standard_spikes!(spiketimes, interval)  # in-place normalize to [0,1] relative time
```

### STTC (`analysis/targets.jl`)

```julia
STTC(A, B, Δt, interval)
# pair → scalar; sorts A and B; calls _tile_fraction + _sttc_pair

STTC(spiketrains::Vector{Vector{Float32}}, Δt, interval)
# N×N matrix; sorts once, pre-computes T[i]; Threads.@threads over pairs

# Internals:
_tile_fraction(st, Δt, istart, iend)  # non-mutating, binary-search-based fraction
_coincident_fraction(A, B, Δt)        # searchsortedfirst, zero allocations
_sttc_pair(A, B, TA, TB, Δt)          # final formula

tile_interval(st, Δt, interval)       # mutating version (called by older code)
is_unimodal(kernel, ratio)            # unimodality test
```

### Population utilities (`analysis/populations.jl`)

```julia
population_indices(P)
# NamedTuple: pop_key → contiguous index range (for slicing activity matrices)

filter_items(P; condition=no_noise)
# filter by name/condition; default excludes pops with "noise" in name

subpopulations(stim, subset=nothing)
# → NamedTuple: stim_name → unique neuron ids it drives

target_neurons(stim, targets::Vector)
# → Vector{Vector{Int}} for named stimuli

average_conn_strength(M, pops::Vector{Vector{Int}}, sparsity=0.2)
# mean(M[post_ids, pre_ids]) / sparsity for all pop pairs → ave_conn matrix
```

---

## 9. Simulation patterns

### Pattern A — point-neuron E/I network

```julia
exc = Population(AdExParameter(); N=800, synapse=DoubleExpSynapse())
pv  = Population(IFParameter();   N=200, synapse=DoubleExpSynapse())
exc_exc = SpikingSynapse(exc, exc, :glu;  conn=(p=0.1f0, μ=1nS))
pv_exc  = SpikingSynapse(pv,  exc, :gaba; conn=(p=0.3f0, μ=2nS))
stim    = PoissonStimulusLayer(200Hz; N=exc.N, conn=(p=1f0, μ=0.5nS))
model   = compose(; exc, pv, exc_exc, pv_exc, stim)
monitor!(model.pop.exc, [:v, :fire])
sim!(model, 2s; dt=0.125ms)
fr, t = firing_rate(model.pop.exc, 0s:1ms:2s; pop_average=true)
```

### Pattern B — Tripod network (dendritic model)

```julia
dend_syn = ReceptorSynapse(Receptors(AMPA, NMDA, GABAa, GABAb), NMDAVoltageDependency())
soma_syn = DoubleExpSynapse()
exc = Population(DendNeuronParameter(ds=[(150um,400um),(150um,400um)]);
                 N=800, adex=AdExParameter(), dend_syn=dend_syn, soma_syn=soma_syn)
th_exc = SpikingSynapse(th, exc, :glu, :d1; conn=(p=0.1f0, μ=1nS))
```

### Pattern C — with STP + LTP

```julia
stp  = MarkramSTPParameter(U=0.3f0, τD=400ms, τF=50ms)
stdp = STDPGerstner(A_pre=0.01, A_post=0.012, τpre=20ms, τpost=20ms)
syn  = SpikingSynapse(exc, exc, :glu; conn=..., STPParam=stp, LTPParam=stdp)
train!(model, 10s)     # plasticity active
```

### Pattern D — warmup → monitor → analyse

```julia
sim!(model, 1s)                    # warmup (no recording)
reset_time!(model)                 # optional: restart time counter
monitor!(model.pop.exc, [:v_s, :fire, :w_s]; sr=1kHz)
sim!(model, 2s; pbar=true)
sp   = spiketimes(model.pop.exc)
fr   = firing_rate(sp; interval=0s:1ms:2s, pop_average=true)
sttc = STTC(sp, 10ms, 0s:1ms:2s)
```

### Pattern E — spatial network

```julia
Npop   = (Exc=800, PV=200)
points = place_populations(Npop, [1.0f0, 1.0f0])
spatial = (type=:gaussian, σs=(Exc=(0.2f0,0.2f0), PV=(0.3f0,0.3f0)),
           ϵ=0.1f0, grid_size=[1.0f0,1.0f0])
L, W, P = compute_connections(:Exc, :Exc, points;
                               conn=(p=0.1f0,), spatial, dist=Normal(1nS, 0.1nS))
syn_ee = SpikingSynapse(exc, exc, :glu; conn=(w=W,))
```

### Pattern F — config-based (STT project)

```julia
base_config = load_parameters()           # Dict from YAML param files
@update! base_config begin
    network.exc.adex.τm = 25ms
    network.exc.adex.b  = 80pA
end
exc = Population(; base_config.network.exc...)
```

---

## 10. Perturbation API (`utils/perturbation.jl`, `utils/copying.jl`)

### `modelcopy` (copying.jl)

Deep copy of a model that replicates Base.deepcopy internals and intercepts `records` dicts: when a dict has a `:data` key, data entries are replaced with empty buffers (monitoring schema preserved, no recorded data).

```julia
ck = modelcopy(model)   # deep copy: same state + aliasing, empty record buffers
```

### `perturbation_test`

```julia
perturbation_test(model, simtime, condition!;
    from_state = nothing,   # pre-built independent model (skips modelcopy)
    add_records = nothing,  # String/Symbol: flush results into model; nothing: return pert model
    train = true,           # true → train!; false → sim!
    kwargs...)
```

Creates `modelcopy(model)` (or uses `from_state`), applies `condition!`, runs sim/train, optionally flushes results into `model`'s perturbation store.

Storage schema:
```
obj.records[:perturbation][variable::Symbol][condition::String][n::Int]
    = Dict("interval" => Float32[tstart, tend],
           "data"     => Matrix{Float32} | Spiketimes)
```

### `perturbation_record`

```julia
v, r = perturbation_record(obj, :v,    condition, interval)  # → Matrix{Float32}, range
st, r = perturbation_record(obj, :fire, condition, interval)  # → Spiketimes, range
```

Returns baseline trace with perturbation windows spliced in. Uses `interpolated_record` for time alignment.

### Clear functions

```julia
clear_perturbation_records!(obj)                          # all perturbation data
clear_perturbation_records!(obj, condition)               # one condition
clear_perturbation_records!(obj, condition; variable, n)  # one variable or n-th recording

clear_perturbation_monitor!(obj, variable)                # by variable
clear_perturbation_monitor!(obj, variable, condition; n)  # by variable + condition
```

### Workflow

```julia
monitor!(model.pop.exc, [:v, :fire]; sr = 1kHz)
ck = modelcopy(model)                   # checkpoint at t=T

# perturbations (both start from t=T)
perturbation_test(ck, 500ms, m -> (m.pop.pv.I .= 200pA); add_records = "pv_drive")
perturbation_test(ck, 500ms, m -> set_active!(m.stim.noise, false); add_records = "no_noise")

# baseline
sim!(ck, 500ms)

t0 = ck.pop.exc.records[:start_time][:v]
t1 = ck.pop.exc.records[:end_time][:v]
v_base, r = perturbation_record(ck.pop.exc, :v, "nonexistent", t0:0.5f0:t1)
v_pert, _ = perturbation_record(ck.pop.exc, :v, "pv_drive",   t0:0.5f0:t1)
```

---

## 11. Open problems and improvement targets

### 10.1 sim! does not call update_traces!

`sim!` inner step calls only `integrate!` + `record!` on populations and `forward!` + `record!` on connections. `update_traces!` is NOT called.
Consequence: any model relying on trace dynamics (STP, STDP, Ca rules) will silently stall if run with `sim!` instead of `train!`. **No documentation of this.**
Fix: add `update_traces=true/false` kwarg to `sim!`, or document clearly.

### 10.2 forward! is single-threaded

`SpikingSynapse.forward!` inner loop is `@inbounds @fastmath @simd` but sequential over pre-neurons.
For N_pre ≈ 1000–5000, this is the hot path in sim!.
Fix candidate: `Threads.@threads` over pre chunks, with per-thread g-buffer to avoid races.

### 10.3 vSTDPVariables.x allocated with Npost, used as pre-trace

`vSTDPVariables` declares `x::VFT = zeros(Npost)` but inside `plasticity!`:
`x[j] += dt*(-x[j]+fireJ[j])/τx` — j is a pre-neuron index.
If Npre ≠ Npost this silently reads/writes wrong memory.
Fix: `x::VFT = zeros(Npre)`.

### 10.4 CaRule.jl is legacy/broken

`CaRule.jl` defines its own duplicate `STDPVariables` struct and dispatches on `c.plasticity`
(not `c.LTPVars`). `SpikingSynapse` has no `plasticity` field — this code is dead.
Either the rule was never ported from an older SpikingSynapse design, or it was replaced by
the STDP_traces.jl rules. **Needs cleanup or port.**

### 10.5 change_plasticity! has undefined variable

`sparse_plasticity.jl:94`: `plasticityvariables(param, Npre, Npost)` — `param` is not defined
in scope (should be `LTP`). Same bug in STP branch.
This function silently fails at runtime whenever called.
Fix: replace `param` with `LTP` / `STP` in the respective branches.

### 10.6 variable_inputs.jl: stimulate! dispatches on undefined type

`stimulate!(p, param::VariablesParameter, ...)` — `VariablesParameter` is not defined anywhere.
Should be `VariableInputParameter`. The stimulus is therefore unreachable via the normal sim loop.

### 10.7 Trace update scheme inconsistent across STDP rules

`STDPGerstner`/`STDPConfavreux2025`: event-driven `exp(-(t-last)/τ)` — traces only updated at spike times.
`STDPMexicanHat`/`iSTDP`/`vSTDP`/`CaRule`/`STDPStructured`: continuous per-dt decay.
Mixed schemes in same codebase → different numerical precision and behaviour under the same model.

### 10.8 Recording allocates on every retrieval

`getvariable` hcat-s the full record list into a new matrix each call.
For 1kHz recording over 10s with N=1000 neurons: O(10M float) allocation per call.
No lazy/streaming access.

### 10.9 firing_rate conv is sequential per neuron

`tmap` parallelizes over neurons via ThreadTools but `conv(spike_train, kernel)` is serial.
For large N × long interval this is the bottleneck.
Fix: `Threads.@threads` over neurons, pre-allocate output buffer.

### 10.10 flush(stdout) in train! hot path

`main.jl:127`: `flush(stdout)` inside every `train!` step. Should be outside the loop.

### 10.11 metaplasticity cadence not configurable

`SynapseNormalization`/`AggregateScaling` normalize every dt if included in syn.
No rate-limiting (every N steps) is exposed — only τ controls timescale.
For large networks normalizing every dt adds significant overhead.

### 10.12 Missing components

- **Gap junctions** — no electrical coupling
- **Dendritic spike / plateau potential** — NMDA via `ReceptorVoltage` but no explicit plateau detection or Ca²⁺ spike nonlinearity in the integrate! loop
- **Per-dendrite spike threshold** — Tripod threshold only at soma
- **Multi-area projection with delay** — `SpikingSynapseDelayParameter` exists but not wired into `compute_connections` or spatial patterns
- **Online metrics** — no per-step hooks for loss, weight histogram, or convergence check
- **GPU support** — no CUDA.jl path; STTC matrix and forward! are natural GPU targets
- **Adaptive time-stepping** — Tripod uses fixed Heun; no multi-rate integration for slow/fast vars

---

## 12. Quick-reference call signatures

```julia
# Build
Population(param; N, synapse, spike, kwargs...)
SpikingSynapse(pre, post, sym, [comp]; conn, LTPParam, STPParam, delay_dist, name)
compose(; pops..., syns..., stims...)

# Run
sim!(model, duration; dt=0.125ms, pbar=false)
train!(model, duration; dt=0.125ms, pbar=false)
get_time(model)    reset_time!(model)

# Record
monitor!(p, syms; sr=1kHz, variables=nothing)
record(p, sym; interval, interpolate=true, range=false)
spiketimes(p; interval)
firing_rate(p, interval; kernel, pop_average, time_average)
clear_records!(model)

# Synapse introspection
matrix(c)             # → SparseMatrix
matrix(c, :W, time)   # → time-sliced SparseMatrix
presynaptic(c, i)     # → pre indices for post i
update_weights!(c, j, i, w)
connect!(c, j, i, μ)

# Perturbation
modelcopy(model)
perturbation_test(model, duration, condition!; from_state, add_records, train)
perturbation_record(obj, :v, condition, interval)  # → (Matrix{Float32}, range)
perturbation_record(obj, :fire, condition, interval)  # → (Spiketimes, range)
clear_perturbation_records!(obj, [condition]; variable, n)
clear_perturbation_monitor!(obj, variable, [condition]; n)

# Analysis
STTC(A, B, Δt, interval)
STTC(spiketrains, Δt, interval)  # → N×N matrix
gaussian_smooth(xs, x, σ)
spatial_activity(points, activity; T, L or N, grid_size)
population_indices(model.pop)

# Spatial
place_populations(Npop, grid_size)
compute_connections(pre, post, points; conn, spatial, dist)
periodic_distance(p1, p2, grid_size)

# Config
load_parameters()      # → Dict from YAML files
@update! config begin field.path = val end
isa_model(model)       # validate structure

# Introspection
print_model(model)
graph(model)           # MetaDiGraph
```
