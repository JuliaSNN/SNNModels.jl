@doc """
Gerstner, W., Kempter, R., van Hemmen, J. L., & Wagner, H. (1996). A neuronal learning rule for sub-millisecond temporal coding. Nature, 383(6595), 76–78. https://doi.org/10.1038/383076a0
"""
STDPGerstner

@snn_kw struct STDPGerstner{FT = Float32} <: STDPParameter
    A_post::FT = 10e-5pA / mV         # LTD learning rate (inhibitory synapses)
    A_pre::FT = 10e-5pA / (mV * mV)  # LTP learning rate (inhibitory synapses)
    τpre::FT = 20ms                   # Time constant for pre-synaptic spike trace
    τpost::FT = 20ms                  # Time constant for post-synaptic spike trace
    Wmax::FT = 30.0pF                 # Max weight
    Wmin::FT = 0.0pF                  # Min weight (negative for inhibition)
end

@snn_kw struct STDPConfavreux2025{FT = Float32} <: STDPParameter
    η::FT = 0.001
    α::FT = 0 ## baseline rate dependency post
    β::FT = 0 ## baseline rate dependency pre
    κ::FT = 1.0f0 # stdp pre->post
    γ::FT = 1.0f0 # stdp post->pre
    τpre::FT = 20ms            
    τpost::FT = 20ms      
    Wmin::FT = 0.0pF
    Wmax::FT = 30.0pF
end

@doc """
    STDPMexicanHat{FT = Float32}
    
    The STDP is defined such that integral of the kernel is zero. The STDP kernel is defined as:

    `` A x * exp(-x/sqrt(2)) ``

    where   ``A`` is the learning rate for post and pre-synaptic spikes, respectively, and ``x`` is the difference between the post and pre-synaptic traces.
"""
STDPMexicanHat

@snn_kw struct STDPMexicanHat{FT = Float32} <: STDPParameter
    A::FT = 10e-2pA / mV    # LTD learning rate (inhibitory synapses)
    τ::FT = 20ms                    # Time constant for pre-synaptic spike trace
    Wmax::FT = 30.0pF                # Max weight
    Wmin::FT = 0.0pF               # Min weight (negative for inhibition)
end

## Common variables for STDP rules
# STDP Variables Structure
@snn_kw struct STDPVariables{VFT = Vector{Float32},IT = Int} <: LTPVariables
    Npost::IT                      # Number of post-synaptic neurons
    Npre::IT                       # Number of pre-synaptic neurons
    tpre::VFT = zeros(Float32, Npre)           # Pre-synaptic spike trace
    tpost::VFT = zeros(Float32, Npost)          # Post-synaptic spike trace
    last_pre::VFT = zeros(Float32, Npre)          # Last pre-synaptic spike time
    last_post::VFT = zeros(Float32,Npost)         # Last post-synaptic spike
    Δpre::VFT = zeros(Float32, length(tpre))
    Δpost::VFT = zeros(Float32, length(tpost))
    active::VBT = [true]
end

# Function to initialize plasticity variables
function plasticityvariables(param::T, Npre, Npost) where {T<:STDPParameter}
    return STDPVariables(Npre = Npre, Npost = Npost)
end

##


# Function to implement STDP update rule
function plasticity!(
    c::PT,
    param::STDPGerstner,
    variables::STDPVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, fireJ, fireI, g, index = c
    @unpack tpre, tpost, last_pre, last_post, Δpre, Δpost = variables
    @unpack A_pre, A_post, τpre, τpost, Wmax, Wmin = param


    # Update weights based on pre-post spike timing
    @inbounds @fastmath begin
        t = get_time(T)
        @simd for j in eachindex(fireJ)
            if fireJ[j]
                tpre[j] = tpre[j] * exp(-(t - last_pre[j]) / τpre) + A_pre
                last_pre[j] = t
            end
            Δpre[j] = t > last_pre[j] ? tpre[j] * exp(-(t - last_pre[j]) / τpre) : 0f0
        end
        @simd for i in eachindex(fireI)
            if fireI[i]
                tpost[i] = tpost[i] * exp(-(t - last_post[i]) / τpost) + A_post
                last_post[i] = t
            end
            Δpost[i] = t > last_post[i] ? tpost[i] * exp(-(t - last_post[i]) / τpost) : 0f0
        end

        chunks = Iterators.partition(eachindex(W), cld(length(W), Threads.nthreads())) |> collect
        Threads.@threads for c in eachindex(chunks) # Iterate over presynaptic neurons
            for s in chunks[c]
                i, j = I[s], J[s] # post and pre neuron indices
                if fireI[i] # post spike
                    W[s] +=  A_pre * Δpre[j] # post-pre
                end
                if fireJ[j] # pre spike
                    W[s] += A_post * Δpost[i] # pre-post
                end
                W[s] = clamp(W[s], Wmin, Wmax)
            end
        end
    end
end

# Function to implement STDP update rule
function plasticity!(
    c::PT,
    param::STDPConfavreux2025,
    variables::STDPVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, fireJ, fireI, g, index = c
    @unpack tpre, tpost, last_pre, last_post, Δpre, Δpost = variables
    @unpack η, α, β, κ, γ, τpre, τpost, Wmin, Wmax = param

    # Update weights based on pre-post spike timing
    # @inbounds 
    @fastmath begin
        t = get_time(T)
        @simd for j in eachindex(fireJ)
            if fireJ[j]
                tpre[j] = tpre[j] * exp(-(t - last_pre[j]) / τpre) + 1f0
                last_pre[j] = t
            end
            Δpre[j] = t > last_pre[j] ? tpre[j] * exp(-(t - last_pre[j]) / τpre) : 0f0
        end
        @simd for i in eachindex(fireI)
            if fireI[i]
                tpost[i] = tpost[i] * exp(-(t - last_post[i]) / τpost) + 1f0
                last_post[i] = t
            end
            Δpost[i] = t > last_post[i] ? tpost[i] * exp(-(t - last_post[i]) / τpost) : 0f0
        end

        # @simd for i = 1:(length(rowptr)-1)
        #     @simd for st = rowptr[i]:(rowptr[i+1]-1)
        #         s = index[st]
        #         if fireJ[J[s]] 
        #             W[s] += η * (κ * Δpost[i] + α) # pre-post
        #             W[s] = clamp(W[s], Wmin, Wmax)
        #         end
        #     end
        # end
        # @simd for j = 1:(length(colptr)-1)
        #     @simd for s = colptr[j]:(colptr[j+1]-1)
        #         if fireI[I[s]] 
        #             W[s] +=  η * (γ * Δpre[j] + β) # post-pre
        #             W[s] = clamp(W[s], Wmin, Wmax)
        #         end
        #     end
        # end

        chunks = Iterators.partition(eachindex(W), cld(length(W), Threads.nthreads())) |> collect
        Threads.@threads for c in eachindex(chunks) # Iterate over presynaptic neurons
            for s in chunks[c]
                i, j = I[s], J[s] # post and pre neuron indices
                if fireI[i] # post spike
                    W[s] +=  η * (γ * Δpre[j] + β) # post-pre
                end
                if fireJ[j] # pre spike
                    W[s] += η * (κ * Δpost[i] + α) # pre-post
                end
                W[s] = clamp(W[s], Wmin, Wmax)
            end
        end
    end
end

MexicanHat(x::Float32) = (1 - x) * exp(-x / sqrt(2)) |> x -> isnan(x) ? 0 : x
function plasticity!(
    c::PT,
    param::STDPMexicanHat,
    plasticity::STDPVariables,
    dt::Float32,
    T::Time,
) where {PT<:AbstractSparseSynapse}
    @unpack rowptr, colptr, I, J, index, W, fireJ, fireI, g, index = c
    @unpack tpre, tpost = plasticity
    @unpack A, τ, Wmax, Wmin = param


    # Update weights based on pre-post spike timing
    @inbounds @fastmath begin

        @turbo for i in eachindex(fireI)
            tpost[i] += dt * (-tpost[i]) / τ
        end
        @simd for i in findall(fireI)
            tpost[i] += 1
        end

        @turbo for j in eachindex(fireJ)
            tpre[j] += dt * (-tpre[j]) / τ
        end
        @simd for j in findall(fireJ)
            tpre[j] += 1
        end


        for i = 1:(length(rowptr)-1)
            @simd for st = rowptr[i]:(rowptr[i+1]-1)
                s = index[st]
                if fireJ[J[s]] && abs(tpost[i] * tpre[J[s]]) > 0.0f0
                    W[s] += A * MexicanHat((log(tpre[J[s]] / tpost[i]))^2)
                end
            end
        end

        # Update weights based on pre-post spike timing
        for j = 1:(length(colptr)-1)
            @simd for s = colptr[j]:(colptr[j+1]-1)
                if fireI[I[s]] && abs(tpost[I[s]] * tpre[j]) > 0.0f0
                    W[s] += A * MexicanHat(log(tpre[j] / tpost[I[s]])^2)
                end
            end
        end
    end
    # Clamp weights to the specified bounds
    @turbo for i in eachindex(W)
        @inbounds W[i] = clamp(W[i], Wmin, Wmax)
    end
end

# Export the relevant functions and structs
export STDPVariables, plasticityvariables, plasticity!, STDPMexicanHat, STDPGerstner, STDPConfavreux2025
