@doc """
    STDPTriplet(; A2_plus = 5.3e-3, A3_plus = 8e-3, A2_minus = 3.5e-3, A3_minus = 0,
                  τ_plus = 16.8ms, τ_minus = 33.7ms, τ_x = 946ms, τ_y = 40ms,
                  Wmax = 30pF, Wmin = 0pF)

Triplet STDP, all-to-all interaction, event-driven
(Pfister, J.-P., & Gerstner, W. (2006). Triplets of spikes in a model of spike
timing-dependent plasticity. J. Neurosci., 26(38), 9673–9682.
https://doi.org/10.1523/JNEUROSCI.1425-06.2006; implementation as Auryn
`MinimalTripletConnection` / `TripletConnection`).

Presynaptic traces ``r_1`` (`τ_plus`), ``r_2`` (`τ_x`); postsynaptic traces
``o_1`` (`τ_minus`), ``o_2`` (`τ_y`). All jump by 1 at a spike and decay
exponentially. Weight updates (amplitudes positive, signs explicit), with every
trace read before this step's spikes are added (the ``t - ε`` of the paper):

- presynaptic spike of j, for each target i:
  ``w_{ij} \\mathrel{-}= o_{1,i} (A_2^- + A_3^- r_{2,j})``
- postsynaptic spike of i, for each source j:
  ``w_{ij} \\mathrel{+}= r_{1,j} (A_2^+ + A_3^+ o_{2,i})``

followed by clamping to `[Wmin, Wmax]`.

Defaults: Table 4 of Pfister & Gerstner (2006), hippocampal culture data set,
all-to-all minimal model (A2_plus = 5.3e-3, A3_plus = 8e-3, A2_minus = 3.5e-3,
A3_minus = 0, τ_y = 40 ms; τ_plus = 16.8 ms and τ_minus = 33.7 ms from Bi & Poo 2001).
These equal Auryn's `MinimalTripletConnection` with `eta = 1`. With `A3_minus = 0`,
`τ_x` is unused; its default (946 ms) is the all-to-all full-model value of Table 4.
Weights are in the same units as `W`; rescale the amplitudes to the weight scale.

Fields: `A2_plus`, `A3_plus`, `A2_minus`, `A3_minus` (amplitudes, all used with the signs
shown above), `τ_plus`, `τ_minus`, `τ_x`, `τ_y` (trace time constants, ms), `Wmax = 30pF`,
`Wmin = 0pF`. State: `STDPTripletVariables`. Event-driven with the ordering of `STDP_kernels.jl`
(same-step pre/post spikes do not interact), applied only under `train!`.

# Example
```julia
using SpikingNeuralNetworks
SNN.@load_units
pre = SNN.Poisson(N = 100, param = SNN.PoissonParameter(10Hz))
post = SNN.IF(N = 10)
rule = SNN.STDPTriplet(A2_plus = 5.3e-3 * 10, A3_plus = 8e-3 * 10, A2_minus = 3.5e-3 * 10)
syn = SNN.SpikingSynapse(pre, post, :ge; conn = (p = 0.1, μ = 10.0), LTPParam = rule)
SNN.train!(model = SNN.compose(; pre, post, syn), duration = 500ms)
```
"""
STDPTriplet

@snn_kw struct STDPTriplet{FT = Float32} <: STDPParameter
    A2_plus::FT = 5.3e-3     # pair LTP amplitude (post spike, pre trace r1)
    A3_plus::FT = 8.0e-3     # triplet LTP amplitude (post spike, r1 * o2)
    A2_minus::FT = 3.5e-3    # pair LTD amplitude (pre spike, post trace o1)
    A3_minus::FT = 0.0       # triplet LTD amplitude (pre spike, o1 * r2)
    τ_plus::FT = 16.8ms      # pre trace r1
    τ_minus::FT = 33.7ms     # post trace o1
    τ_x::FT = 946ms          # pre trace r2 (triplet LTD)
    τ_y::FT = 40ms           # post trace o2 (triplet LTP)
    Wmax::FT = 30.0pF        # Max weight
    Wmin::FT = 0.0pF         # Min weight
end

@doc """
    STDPTripletVariables(; Npre, Npost, r1 = zeros(Npre), r2 = zeros(Npre), o1 = zeros(Npost),
                         o2 = zeros(Npost), last_pre = zeros(Npre), last_post = zeros(Npost),
                         initialized = [false], active = [true], scale = ones(4),
                         decay = fill(NaN, 9))

Traces of `STDPTriplet`: `r1`, `r2` (presynaptic, length Npre), `o1`, `o2` (postsynaptic, length
Npost); last spike times `last_pre`, `last_post` (informational); `initialized` (one-off clamp done);
`active` (the rule runs only if `any(active)`).

!!! warning "The stored traces are not the trace values"
    As for `STDPVariables`: the value of `r1`, `r2`, `o1`, `o2` is the stored vector times `scale[1:4]`.
    Read values with [`trace_values`](@ref) or call [`sync_traces!`](@ref) first.

`decay` caches `dt`, the time constants and their decay factors. Created by
`plasticityvariables(::STDPTriplet, Npre, Npost)`.
"""
STDPTripletVariables

@snn_kw struct STDPTripletVariables{VFT = Vector{Float32},IT = Int,VDT = Vector{Float64}} <: LTPVariables
    Npost::IT
    Npre::IT
    r1::VFT = zeros(Float32, Npre)          # STORED pre trace, τ_plus (value = r1 * scale[1])
    r2::VFT = zeros(Float32, Npre)          # pre trace, τ_x (times scale[2])
    o1::VFT = zeros(Float32, Npost)         # post trace, τ_minus (times scale[3])
    o2::VFT = zeros(Float32, Npost)         # post trace, τ_y (times scale[4])
    last_pre::VFT = zeros(Float32, Npre)    # Last pre-synaptic spike time
    last_post::VFT = zeros(Float32, Npost)  # Last post-synaptic spike time
    initialized::VBT = [false]
    active::VBT = [true]
    scale::VDT = ones(4)        # factors of the stored traces (r1, r2, o1, o2)
    decay::VDT = fill(NaN, 9)   # cached dt, τs and decay factors (_decay_factors!)
end

function plasticityvariables(param::STDPTriplet, Npre, Npost)
    return STDPTripletVariables(Npre = Npre, Npost = Npost, VFT = _vft(param))
end

function sync_traces!(v::STDPTripletVariables)
    for (x, k) in ((v.r1, 1), (v.r2, 2), (v.o1, 3), (v.o2, 4))
        v.scale[k] == 1.0 || (_rescale!(x, eltype(x)(v.scale[k])); v.scale[k] = 1.0)
    end
    return v
end

function trace_values(v::STDPTripletVariables, key::Symbol)
    x = getfield(v, key)
    k = key === :r1 ? 1 : key === :r2 ? 2 : key === :o1 ? 3 : key === :o2 ? 4 : 0
    return (k == 0 || v.scale[k] == 1.0) ? x : x .* eltype(x)(v.scale[k])
end

lazy_variables(::STDPTripletVariables) = (:r1, :r2, :o1, :o2)

current_value(v::STDPTripletVariables, param, key::Symbol, t::Real) = trace_values(v, key)

# Event-driven triplet STDP; ordering as in STDP_kernels.jl.
function plasticity!(
    c::PT,
    param::STDPTriplet,
    variables::STDPTripletVariables,
    dt::AbstractFloat,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W = c
    @unpack r1, r2, o1, o2, last_pre, last_post, initialized, scale, decay = variables
    @unpack A2_plus, A3_plus, A2_minus, A3_minus, τ_plus, τ_minus, τ_x, τ_y, Wmax, Wmin = param
    t = get_time(T)
    # a new dt or τ: bring the stored traces to their values before the factors change
    _decay_stale(decay, dt, τ_plus, τ_x, τ_minus, τ_y) && sync_traces!(variables)
    f_r1, f_r2, f_o1, f_o2 = _decay_factors!(decay, dt, τ_plus, τ_x, τ_minus, τ_y)
    spikes_pre, spikes_post = presynaptic_spikes(c), postsynaptic_spikes(c)
    _initial_clamp!(W, initialized, Wmin, Wmax)
    # value = stored * scale; these are the values at the start of this step
    s_r1, s_r2, s_o1, s_o2 = _read_scale(scale, 1, r1), _read_scale(scale, 2, r2), _read_scale(scale, 3, o1), _read_scale(scale, 4, o2)

    # 1. pre spike of j onto i: LTD, o1[i] * (A2- + A3- * r2[j]) (r2 before increment)
    f_pre = (w, i, j) -> -(o1[i] * s_o1) * (A2_minus + A3_minus * (r2[j] * s_r2))
    # 2. post spike of i from j: LTP, r1[j] * (A2+ + A3+ * o2[i]) (o2 before increment)
    f_post = (w, i, j) -> (r1[j] * s_r1) * (A2_plus + A3_plus * (o2[i] * s_o2))
    _stdp_passes!(c, f_pre, f_post, spikes_pre, spikes_post, Wmin, Wmax)

    # 3. trace increments (last spike time stored once per neuron)
    if !isempty(spikes_pre)
        _spike_increment!(r1, last_pre, spikes_pre, t, _write_inc(scale, 1, r1))
        _spike_increment!(r2, last_pre, spikes_pre, t, _write_inc(scale, 2, r2))
    end
    if !isempty(spikes_post)
        _spike_increment!(o1, last_post, spikes_post, t, _write_inc(scale, 3, o1))
        _spike_increment!(o2, last_post, spikes_post, t, _write_inc(scale, 4, o2))
    end
    # 4. exact decay over one step
    _advance_scale!(scale, 1, r1, f_r1)
    _advance_scale!(scale, 2, r2, f_r2)
    _advance_scale!(scale, 3, o1, f_o1)
    _advance_scale!(scale, 4, o2, f_o2)
    return nothing
end

export STDPTriplet, STDPTripletVariables
