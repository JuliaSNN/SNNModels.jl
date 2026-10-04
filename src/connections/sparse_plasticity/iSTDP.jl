"""
    iSTDPParameter <: STDPParameter

Abstract supertype of the inhibitory STDP rules of Vogels et al. (2011): `iSTDPRate`,
`iSTDPPotential` and `iSTDPTime`. All of them use `iSTDPVariables` and
Euler-integrated (not exact-decay) traces. See the notes on `iSTDPRate`.
"""
abstract type iSTDPParameter <: STDPParameter end

@doc """
    iSTDPRate{FT = Float32}

Inhibitory STDP with a target postsynaptic rate (Vogels, T. P., Sprekeler, H.,
Zenke, F., Clopath, C., & Gerstner, W. (2011). Inhibitory plasticity balances excitation
and inhibition in sensory pathways and memory networks. Science, 334(6062), 1569-1573.
https://doi.org/10.1126/science.1211095).

Fields (weights in the units of `W`, e.g. pF):
- `η = 0.01pA`: learning rate.
- `r = 3Hz`: target postsynaptic rate.
- `τy = 50ms`: time constant of the pre- and postsynaptic traces.
- `Wmax = 243pF`, `Wmin = 0.01pF`: weight bounds.

Algorithm (per step, `tpre`, `tpost` are Euler-integrated traces that jump by 1 at a spike;
`I[s]` is the postsynaptic and `J[s]` the presynaptic index of synapse `s`):
- presynaptic spike of `j`, for every outgoing synapse `s` (post `i = I[s]`):
  `W[s] += η * (tpost[i] - 2 r τy)`. Depression is the constant term `-2 η r τy`.
- postsynaptic spike of `i`, for every incoming synapse `s` (pre `j = J[s]`):
  `W[s] += η * tpre[j]`.
- every weight touched is clamped to `[Wmin, Wmax]`.

Unlike the event-driven pair rules (`STDPGerstner` etc.), traces are Euler-integrated and a
spike is added to its trace before the postsynaptic pass of the same step reads `tpre`, so
a pre and a post spike in the same step do interact.

Plasticity is applied only when the network is run with `train!`; `sim!` never calls it.

!!! warning "Bug fixed in SNNModels 1.9"
    In SNNModels 1.5.0 - 1.8.1 (and SpikingNeuralNetworks.jl from v1.0.0, commit 680a30c) the
    potentiation applied at a postsynaptic spike used a `@turbo` loop that reassigned its loop
    variable. LoopVectorization ignored the reassignment, so the update hit the synapses stored
    at CSC positions `rowptr[i]:rowptr[i+1]-1` (synapses onto unrelated postsynaptic neurons)
    instead of the incoming synapses of the spiking neuron `i`. The depression term was correct.
    Simulations that used `iSTDPRate` with those versions must be rerun.

# Example
```julia
E = SNN.IF(N = 800); I = SNN.IF(N = 200)
IE = SNN.SpikingSynapse(I, E, :gi; conn = (p = 0.2, μ = 5.0), LTPParam = SNN.iSTDPRate(r = 5Hz))
SNN.train!(model = SNN.compose(; E, I, IE), duration = 5s)   # plasticity needs train!
```
"""
iSTDPRate

@snn_kw struct iSTDPRate{FT = Float32} <: iSTDPParameter
    η::FT = 0.01pA
    r::FT = 3Hz
    τy::FT = 50ms
    Wmax::FT = 243pF
    Wmin::FT = 0.01pF
end

@doc """
    iSTDPTime{FT = Float32}

Parameter container (`η`, `τy`, `Wmax`, `Wmin`) of a time-based variant of the Vogels et al.
(2011) rule. The current code defines no `plasticity!` method for it, so it cannot be used as
`LTPParam` (the update was removed in commit e4ce94f, 2025-12-24; before that it contained the
same post-spike `@turbo` bug described in `iSTDPRate`). Use `iSTDPRate` or `iSTDPPotential`.
"""
iSTDPTime

@snn_kw struct iSTDPTime{FT = Float32} <: iSTDPParameter
    η::FT = 0.01pA
    τy::FT = 50ms
    Wmax::FT = 243pF
    Wmin::FT = 0.01pF
end

@doc """
    iSTDPPotential{FT = Float32}

Inhibitory STDP in which the postsynaptic trace follows the membrane potential instead of the
spike train (Vogels et al. 2011 form, with `tpost` low-pass filtering `v_post`).

Fields: `η = 0.001pA` (learning rate), `v0 = -50mV` (reference potential),
`τy = 200ms` (trace time constant), `Wmax = 243pF`, `Wmin = 0.01pF`.

Algorithm (per step; `tpre` is an Euler-integrated spike trace, `tpost` an Euler-integrated
trace of `v_post`):
- presynaptic spike of `j`, every outgoing synapse `s` (post `i = I[s]`):
  `W[s] += η * (tpost[i] - v0)`.
- postsynaptic spike of `i`, every incoming synapse `s` (pre `j = J[s]`):
  `W[s] += η * tpre[j]`.
- every weight touched is clamped to `[Wmin, Wmax]`.

Plasticity is applied only under `train!`. This rule was not affected by the `iSTDPRate` bug
fixed in SNNModels 1.9 (its postsynaptic loop never used `@turbo`); it now uses the same plain
`@simd` loops.
"""
iSTDPPotential

@snn_kw mutable struct iSTDPPotential{FT = Float32} <: iSTDPParameter
    η::FT = 0.001pA
    v0::FT = -50mV
    τy::FT = 200ms
    Wmax::FT = 243pF
    Wmin::FT = 0.01pF
end


@doc """
    iSTDPVariables

State of the inhibitory STDP rules: `tpre` (length `Npre`) and `tpost` (length `Npost`)
traces, `last_spike` (unused by the update) and the `active` flag. Record the traces with
`LTPVars` as the variable set, e.g. `monitor!(syn, [:tpost], :LTPVars)`.
"""
iSTDPVariables

@snn_kw struct iSTDPVariables{VFT = Vector{Float32},IT = Int} <: PlasticityVariables
    ## Plasticity variables
    Npost::IT
    Npre::IT
    tpost::VFT = zeros(Npost) # postsynaptic spiking time 
    tpre::VFT = zeros(Npre) # presynaptic spiking time
    last_spike::VFT = zeros(Npost) # last spike time for each postsynaptic neuron
    active::VBT = [true]
end

function plasticityvariables(param::T, Npre, Npost) where {T<:iSTDPParameter}
    return iSTDPVariables(Npre = Npre, Npost = Npost)
end

"""
    plasticity!(c::AbstractSparseSynapse, param::iSTDPRate, variables::iSTDPVariables, dt::Float32, T::Time)

One step of the inhibitory STDP of Vogels et al. (2011) with target rate `param.r`; modifies
`c.W` in place. Called by `train!` (never by `sim!`). See `iSTDPRate` for the update equations.

# Arguments
- `c`: the synapse; uses `rowptr`, `colptr`, `index`, `I`, `J`, `W`, `fireI` (post spikes), `fireJ` (pre spikes).
- `param::iSTDPRate`: `η`, `r`, `τy`, `Wmin`, `Wmax`.
- `variables::iSTDPVariables`: traces `tpre`, `tpost`, updated in place.
- `dt::Float32`: time step (ms); `T::Time`: simulation clock (unused).

# Algorithm
- Every neuron's trace is decayed by an Euler step `-dt * trace / τy`; a spike adds 1.
- Presynaptic spike: walk the outgoing synapses (`colptr`) and add `η (tpost[i] - 2 r τy)`.
- Postsynaptic spike `i`: walk the incoming synapses `k = rowptr[i]:rowptr[i+1]-1`, map to the
  CSC position `s = index[k]` and add `η tpre[J[s]]`. Weights are clamped to `[Wmin, Wmax]`.
"""
function plasticity!(
    c::AbstractSparseSynapse,
    param::iSTDPRate,
    plasticity::iSTDPVariables,
    dt::Float32,
    T::Time,
)
    @unpack rowptr, colptr, index, I, J, W, fireI, fireJ, g = c
    @unpack η, r, τy, Wmax, Wmin = param
    @unpack tpre, tpost = plasticity
    # @inbounds 
    # if pre-synaptic inhibitory neuron fires, it aims to modify the synaptic weight to achieve the target post synaptic rate
    @fastmath begin
        @inbounds for j in eachindex(fireJ) # presynaptic indices j
            tpre[j] += dt * (-tpre[j]) / τy
            if fireJ[j] # presynaptic neuron
                tpre[j] += 1
                @simd for s = colptr[j]:(colptr[j+1]-1)
                    W[s] = clamp(W[s] + η * (tpost[I[s]] - 2 * r * τy), Wmin, Wmax)
                end
            end
        end
        # if post-synaptic excitatory neuron fires
        # @inbounds 
        @inbounds for i in eachindex(fireI) # postsynaptic indices i
            tpost[i] += dt * (-tpost[i]) / τy
            if fireI[i] # postsynaptic neuron
                tpost[i] += 1
                # k walks the row-ordered view; s = index[k] is the CSC position.
                # (Plain loop: the former @turbo reassigned its loop variable, which
                # LoopVectorization does not support.)
                @simd for k = rowptr[i]:(rowptr[i+1]-1)
                    s = index[k]
                    W[s] = clamp(W[s] + η * tpre[J[s]], Wmin, Wmax)
                end
            end
        end
    end
end

"""
    plasticity!(c::AbstractSparseSynapse, param::iSTDPPotential, variables::iSTDPVariables, dt::Float32, T::Time)

Performs the synaptic plasticity calculation based on the inihibitory spike-timing dependent plasticity (iSTDP) model from Vogels (2011) using membrane potential traces.
The function updates synaptic weights `W` of each synapse in the network according to the firing status of pre and post-synaptic neurons.
This is an in-place operation that modifies the input `AbstractSparseSynapse` object `c`.

# Arguments
- `c::AbstractSparseSynapse`: The current spiking synapse object which contains data structures to represent the synapse network.
- `param::iSTDPPotential`: Parameters needed for the iSTDP model, including learning rate `η`, reference membrane potential `v0`, STDP time constant `τy`, maximal and minimal synaptic weight (`Wmax` and `Wmin`).
- `dt::Float32`: The time step for the numerical integration. 

# Algorithm
- For each pre-synaptic neuron, if it fires, it increases the synaptic weight by an amount proportional to the difference between the post-synaptic membrane potential trace and a reference potential `v0`, otherwise the pre-synaptic trace decays exponentially over time with a time constant `τy`.
- For each post-synaptic neuron, if it fires, it increases the synaptic weight by an amount proportional to the pre-synaptic trace, otherwise the post-synaptic trace decays exponentially over time with a time constant `τy`.
- The synaptic weights are bounded by `Wmin` and `Wmax`.

"""
function plasticity!(
    c::AbstractSparseSynapse,
    param::iSTDPPotential,
    plasticity::iSTDPVariables,
    dt::Float32,
    T::Time,
)
    @unpack rowptr, colptr, index, I, J, W, v_post, fireI, fireJ, g = c
    @unpack η, v0, τy, Wmax, Wmin = param
    @unpack tpre, tpost = plasticity

    # @inbounds 
    # if pre-synaptic inhibitory neuron fires
    @fastmath @inbounds for j in eachindex(fireJ) # presynaptic indices j
        tpre[j] += dt * (-tpre[j]) / τy
        if fireJ[j] # presynaptic neuron
            tpre[j] += 1
            @simd for s = colptr[j]:(colptr[j+1]-1)
                W[s] = clamp(W[s] + η * (tpost[I[s]] - v0), Wmin, Wmax)
            end
        end
    end

    # if post-synaptic excitatory neuron fires
    @fastmath @inbounds for i in eachindex(fireI) # postsynaptic indices i
        # trace of the membrane potential
        tpost[i] += dt * -(tpost[i] - v_post[i]) / τy
        if fireI[i] # postsynaptic neuron
            @simd for k = rowptr[i]:(rowptr[i+1]-1)
                s = index[k]
                W[s] = clamp(W[s] + η * tpre[J[s]], Wmin, Wmax)
            end
        end
    end
end

export iSTDPRate,
    iSTDPTime, iSTDPPotential, iSTDPVariables, plasticityvariables, plasticity!
