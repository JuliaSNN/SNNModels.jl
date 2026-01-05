
"""
    SpikingSynapseParameter <: AbstractConnectionParameter
"""
struct SpikingSynapseParameter <: AbstractSpikingSynapseParameter end

@snn_kw struct SpikingSynapseDelayParameter{VVFT =Vector{Vector{Float32}}, VFT = Vector{Float32}} <: AbstractSpikingSynapseParameter
    delaytime::VFT
    spike_time::VVFT
    spike_w::VVFT
end

@snn_kw mutable struct SpikingSynapse{
                VIT = Vector{Int32},
                VFT = Vector{Float32},
                SYNP <: AbstractSpikingSynapseParameter
                } <: AbstractSpikingSynapse
    id::String = randstring(12)
    name::String = "SpikingSynapse"
    param::SYNP = SpikingSynapseParameter()
    LTPParam::LTPParameter = NoLTP()
    STPParam::STPParameter = NoSTP()
    LTPVars::PlasticityVariables = NoPlasticityVariables()
    STPVars::PlasticityVariables = NoPlasticityVariables()
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::VFT  # synaptic weight
    ρ::VFT  # short-term plasticity
    fireI::VBT # postsynaptic firing
    fireJ::VBT # presynaptic firing
    v_post::VFT
    g::VFT  # rise conductance
    targets::Dict = Dict()
    records::Dict = Dict()
end

"""
    SpikingSynapse to connect neuronal populations
"""
SpikingSynapse

function SpikingSynapse(
    pre::AbstractPopulation,
    post::AbstractPopulation,
    sym::Symbol,
    comp::Union{Symbol,Nothing} = nothing;
    conn::Connectivity,
    delay_dist::Union{Distribution,Nothing} = nothing,
    dt::Float32 = 0.125f0,
    LTPParam::LTPParameter = NoLTP(),
    STPParam::STPParameter = NoSTP(),
    name::String = "SpikingSynapse",
)

    # set the synaptic weight matrix
    w = sparse_matrix(pre.N, post.N, conn)
    # remove autapses if pre == post
    (pre == post) && (w[diagind(w)] .= 0)
    # get the sparse representation of the synaptic weight matrix
    rowptr, colptr, I, J, index, W = dsparse(w)
    # get the presynaptic and postsynaptic firing
    fireI, fireJ = post.fire, pre.fire

    # get the conductance and membrane potential of the target compartment if multicompartment model
    targets = Dict{Symbol,Any}(
        :fire => pre.id,
        :post => post.id,
        :pre => pre.id,
        :type=>:SpikingSynapse,
    )
    @views g, v_post = synaptic_target(targets, post, sym, comp)

    # set the paramter for the synaptic plasticity
    LTPVars = plasticityvariables(LTPParam, pre.N, post.N)
    STPVars = plasticityvariables(STPParam, pre.N, post.N)

    # short term plasticity
    ρ = copy(W)
    ρ .= 1.0

    # Network targets

    if isnothing(delay_dist)
        param = SpikingSynapseParameter()
    else
        delaytime = rand(delay_dist, length(W))
        spike_time = [[] for _ in 1:length(fireI)]
        spike_w = [[] for _ in 1:length(fireI)]
        param = SpikingSynapseDelayParameter(;
            delaytime,
            spike_time,
            spike_w,
        )
    end
    return SpikingSynapse(;
            ρ = ρ,
            param = param,
            g = g,
            targets = targets,
            @symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, v_post)...,
            LTPVars,
            STPVars,
            LTPParam,
            STPParam,
            name,
        )   
end

function update_plasticity!(c::SpikingSynapse; LTP = nothing, STP = nothing)
    if !isnothing(LTP)
        c.LTPParam = LTP
        c.LTPVars = plasticityvariables(c.LTPParam, length(c.fireJ), length(c.fireI))
    end
    if !isnothing(STP)
        c.STPParam = STP
        c.STPVars = plasticityvariables(c.STPParam, length(c.fireJ), length(c.fireI))
    end
end


function forward!(c::SpikingSynapse, param::SpikingSynapseParameter, dt::Float32, T::Time)
    @unpack colptr, I, W, fireJ, g, ρ = c
    @inbounds for j ∈ eachindex(fireJ) # loop on presynaptic neurons
        if fireJ[j] # presynaptic fire
            @inbounds @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                g[I[s]] += W[s] * ρ[s]
            end
        end
    end
end



function forward!(c::SpikingSynapse, param::SpikingSynapseDelayParameter, dt::Float32, T::Time)
    @unpack colptr, I, W, fireJ, fireI, g, ρ = c
    @unpack delaytime, spike_time, spike_w = param

    for j ∈ eachindex(fireJ) # loop on presynaptic neurons
        if fireJ[j] # presynaptic fire
            @inbounds @fastmath @simd for s ∈ colptr[j]:(colptr[j+1]-1)
                i = I[s]
                times = spike_time[i]
                weights = spike_w[i]
                spike = get_time(T) + delaytime[s]
                first_spike_id = findlast(.<(spike), times)
                first_spike_id = first_spike_id === nothing ? 0 : first_spike_id
                insert!(times, first_spike_id+1, spike)
                insert!(weights, first_spike_id+1, W[s] * ρ[s])
            end
        end
    end
    @fastmath @inbounds @simd for i ∈ eachindex(fireI)
        if !isempty(spike_time[i])
            times = spike_time[i]
            weights = spike_w[i]
            while !isempty(times) && times[1] <= get_time(T)
                g[i] += weights[1]
                popfirst!(times)
                popfirst!(weights)
            end
        end
    end
end


export SpikingSynapse, SpikingSynapseDelay, update_plasticity!
