@doc """
    STDPTriplet{FT = Float32}

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
    STDPTripletVariables

Traces of `STDPTriplet`: `r1`, `r2` (presynaptic, length Npre), `o1`, `o2`
(postsynaptic, length Npost), decayed by `exp(-dt/τ)` every step; last spike times
`last_pre`, `last_post` (informational); `initialized` (one-off clamp done).
"""
STDPTripletVariables

@snn_kw struct STDPTripletVariables{VFT = Vector{Float32},IT = Int} <: LTPVariables
    Npost::IT
    Npre::IT
    r1::VFT = zeros(Float32, Npre)          # pre trace, τ_plus
    r2::VFT = zeros(Float32, Npre)          # pre trace, τ_x
    o1::VFT = zeros(Float32, Npost)         # post trace, τ_minus
    o2::VFT = zeros(Float32, Npost)         # post trace, τ_y
    last_pre::VFT = zeros(Float32, Npre)    # Last pre-synaptic spike time
    last_post::VFT = zeros(Float32, Npost)  # Last post-synaptic spike time
    initialized::VBT = [false]
    active::VBT = [true]
end

function plasticityvariables(param::STDPTriplet, Npre, Npost)
    return STDPTripletVariables(Npre = Npre, Npost = Npost)
end

# Event-driven triplet STDP; ordering as in STDP_kernels.jl.
function plasticity!(
    c::PT,
    param::STDPTriplet,
    variables::STDPTripletVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, fireJ, fireI = c
    @unpack r1, r2, o1, o2, last_pre, last_post, initialized = variables
    @unpack A2_plus, A3_plus, A2_minus, A3_minus, τ_plus, τ_minus, τ_x, τ_y, Wmax, Wmin = param
    t = get_time(T)
    _initial_clamp!(W, initialized, Wmin, Wmax)

    # 1. pre spike of j onto i: LTD, o1[i] * (A2- + A3- * r2[j]) (r2 before increment)
    f_pre = (w, i, j) -> -o1[i] * (A2_minus + A3_minus * r2[j])
    _pre_spike_pass!(f_pre, W, colptr, I, fireJ, Wmin, Wmax)
    # 2. post spike of i from j: LTP, r1[j] * (A2+ + A3+ * o2[i]) (o2 before increment)
    f_post = (w, i, j) -> r1[j] * (A2_plus + A3_plus * o2[i])
    _post_spike_pass!(f_post, W, rowptr, index, J, fireI, Wmin, Wmax)

    # 3. trace increments (last spike time stored once per neuron)
    _spike_increment!(r1, last_pre, fireJ, t)
    _spike_increment!(r2, last_pre, fireJ, t)
    _spike_increment!(o1, last_post, fireI, t)
    _spike_increment!(o2, last_post, fireI, t)
    # 4. exact decay over one step
    _decay!(r1, Float32(exp(-dt / τ_plus)))
    _decay!(r2, Float32(exp(-dt / τ_x)))
    _decay!(o1, Float32(exp(-dt / τ_minus)))
    _decay!(o2, Float32(exp(-dt / τ_y)))
    return nothing
end

export STDPTriplet, STDPTripletVariables
