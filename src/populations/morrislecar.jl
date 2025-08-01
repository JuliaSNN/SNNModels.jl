# Parameters from : https://github.com/nest/ode-toolbox/blob/master/tests/morris_lecar.json
@snn_kw struct MorrisLecarParameter{FT = Float32}
    Cm::FT = 6.69pF
    El::FT = -50mV
    EK::FT = -70mV
    ECa::FT = 100mV
    gl::FT = 0.5nS
    gK::FT = 2nS
    gCa::FT = 1.1nS
    τe::FT = 5ms
    τi::FT = 10ms
    V1::FT = 30mV
    V2::FT = 15mV
    V3::FT = 0mV
    V4::FT = 30mV
    ϕ::FT  = 25Hz
    Ee::FT = 0mV
    Ei::FT = -75mV
end


@snn_kw mutable struct MorrisLecar{VFT = Vector{Float32},VBT = Vector{Bool}} <: AbstractPopulation
    name::String = "MorrisLecar"
    id::String = randstring(12)
    param::MorrisLecarParameter = MorrisLecarParameter()
    N::Int32 = 100
    v::VFT = -52.14 .+ zeros(N)
    w::VFT = 0.2 .+ zeros(N)
    ge::VFT = zeros(N)
    gi::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
end

"""
[Morris-Lecar Neuron](https://www.cell.com/biophysj/pdf/S0006-3495(81)84782-0.pdf?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0006349581847820%3Fshowall%3Dtrue)
"""
MorrisLecar

function integrate!(p::MorrisLecar, param::MorrisLecarParameter, dt::Float32)
    @unpack N, v, w, ge, gi, fire, I = p
    @unpack Cm, El, EK, ECa, gl, gK, gCa, τe, τi, V1, V2, V3, V4, ϕ = param
    @unpack Ee, Ei = param
    @inbounds for i = 1:N
        m_ss = 0.5*(1+tanh((v[i]-V1)/V2))
        n_ss = 0.5*(1+tanh((v[i]-V3)/V4))
        τ = 1 / (ϕ  * cosh((v[i]-V3)/(2V4)))

        # v[i] += dt / Cm * (
        #         I[i] +
        #         gl  * (El  - v[i]) +
        #         gCa * (ECa - v[i]) * m_ss +
        #         gK  * (EK  - v[i]) * w[i] +
        #         0
        #     )
        # w[i]  += dt * (n_ss - w[i])/τ

        v[i] += dt / Cm * MorrisLecar_dv(v[i], w[i], I[i], param)
        w[i] += dt * MorrisLecar_dw(v[i], w[i], param)

        v[i] += dt/Cm * (
                ge[i] * (Ee - v[i]) +
                gi[i] * (Ei - v[i]) 
        )
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
    end
    @inbounds for i = 1:N
        fire[i] = v[i] > 20.0f0

    end
end


function MorrisLecar_dv(v::Float32, w::Float32, I::Float32, param::MorrisLecarParameter)
    @unpack Cm, El, EK, ECa, gl, gK, gCa, τe, τi, V1, V2, V3, V4, ϕ = param
    m_ss = 0.5*(1+tanh((v-V1)/V2))
    return  I +
            gl  * (El  - v) +
            gCa * (ECa - v) * m_ss +
            gK  * (EK  - v) * w
    return dv
end


function MorrisLecar_dw(v::Float32, w::Float32, param::MorrisLecarParameter)  
    @unpack Cm, El, EK, ECa, gl, gK, gCa, τe, τi, V1, V2, V3, V4, ϕ = param
    n_ss = 0.5*(1+tanh((v-V3)/V4))
    τ = 1 / (ϕ * cosh((v-V3)/(2V4)))
    return (n_ss - w)/τ
end


function MorrisLecar_w_nullcline(v::Float32, param::MorrisLecarParameter)  
    @unpack Cm, El, EK, ECa, gl, gK, gCa, τe, τi, V1, V2, V3, V4, ϕ = param
    n_ss = 0.5*(1+tanh((v-V3)/V4))
    return -n_ss
end


function MorrisLecar_v_nullcline(v::Float32, I::Float32, param::MorrisLecarParameter)
    @unpack Cm, El, EK, ECa, gl, gK, gCa, τe, τi, V1, V2, V3, V4, ϕ = param
    m_ss = 0.5*(1+tanh((v-V1)/V2))
    return  -(I +
            gl  * (El  - v) +
            gCa * (ECa - v) * m_ss)/(
            gK  * (EK  - v))
    return dv
end

function plasticity!(p::MorrisLecar, param::MorrisLecarParameter, dt::Float32, T::Time )
end

export MorrisLecar

# function HH_spike_count(p::HH, dt = 0.01)
#     neurons = hcat(p.records[:fire]...)
#     spike_count = zeros(size(neurons, 1))
#     for (n, fires) in enumerate(eachrow(neurons))
#         r = length(findall(x -> fires[x] > 0 && fires[x+1] == 0, eachindex(fires[1:end-1])))
#         spike_count[n] = r / 1000
#     end
#     return spike_count
# end
