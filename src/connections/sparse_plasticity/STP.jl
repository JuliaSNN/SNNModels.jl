"""
    MarkramSTPParameter{FT <: AbstractFloat} <: STPParameter

    The model is based on refractoriness of the synaptic release process, which can be rephrased by stating that:
    The fraction (U) of the synaptic efficacy used by an AP becomes instantaneously unavailable for subsequent use and recovers with a time constant of τD (τrec, depression). The fraction of available synaptic efficacy is termed `x.`  A facilitating mechanism is included in the model as a pulsed increase in U by each AP. The running value of U is referred to as u and U remains a parameter that applies to the first AP in a train. u decays with a single exponential, τF (facilitation), to its resting value U. The amount of synaptic efficacy enhanced by a action potential is assumed to be U(1-u).
    The increase in the amplitude of the postsynaptic response is proportional to the product of u and x.

    The actual implementation follows the equations described in Mongillo et al. (2008) for clarity.

# Fields
- `τD::FT`: Time constant for depression (default: 200ms)
- `τF::FT`: Time constant for facilitation (default: 1500ms)
- `U::FT`: Maximum utilization of synaptic resources (default: 0.2)
- `Wmax::FT`: Maximum synaptic weight (default: 1.0pF)
- `Wmin::FT`: Minimum synaptic weight (default: 0.0pF)

This struct is used to configure the short-term plasticity dynamics in synaptic connections
following the model described by Markram et al. (1998).
"""
MarkramSTPParameter
@snn_kw struct MarkramSTPParameter{FT = Float32} <: STPParameter
    τD::FT = 200ms # τx
    τF::FT = 1500ms # τu
    U::FT = 0.2
    Wmax::FT = 1.0pF
    Wmin::FT = 0.0pF
end

@snn_kw struct MarkramSTPVariables{VFT = Vector{Float32},IT = Int} <: STPVariables
    ## Plasticity variables
    Npost::IT
    Npre::IT
    u::VFT = zeros(Npre) # presynaptic state
    x::VFT = ones(Npre)  # presynaptic state
    _ρ::VFT = ones(Npre) # presynaptic state
    last_spike::VFT = fill(-Inf, Npre)
    active::VBT = [true]
end

function plasticityvariables(param::MarkramSTPParameter, Npre, Npost)
    variables = MarkramSTPVariables(Npre = Npre, Npost = Npost)
    ## initialize variables
    variables.u .= param.U
    variables.x .= 1.0
    variables._ρ .= param.U
    return variables
end


function plasticity!(
    c::PT,
    param::MarkramSTPParameter,
    plasticity::MarkramSTPVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, v_post, fireJ, g, ρ, index = c
    @unpack u, x, _ρ = plasticity
    @unpack U, τF, τD, Wmax, Wmin = param

    @fastmath @inbounds @simd for j in eachindex(fireJ) # Iterate over all columns, j: presynaptic neuron
        if fireJ[j]
            ΔT = get_time(T) - plasticity.last_spike[j]
            plasticity.last_spike[j] = get_time(T)
            # update u and x based on time since last spike
            begin
                u[j] = U + (U - u[j]) * exp64(-ΔT / τF) 
                x[j] = 1 + (1 - x[j]) * exp64(-ΔT / τD)
                u[j] += U * (1 - u[j])
                x[j] += (-u[j] * x[j])
                _ρ[j] = u[j] * x[j]
            end
        end
    end

    @simd for j in eachindex(fireJ) # Iterate over postsynaptic neurons
        @turbo for s = colptr[j]:(colptr[j+1]-1)
            ρ[s] = _ρ[j]
        end
    end
end

export MarkramSTPParameter, MarkramSTPVariables, plasticityvariables, plasticity!
