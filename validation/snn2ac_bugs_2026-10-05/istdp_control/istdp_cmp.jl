# Run identically in the fixed (registered 1.8.2) and buggy (control branch) environments.
using SNNModels, Random, JLD2, Statistics
@load_units
label = ARGS_LABEL

# --- 1. kernel level: same synapse, same forced spikes ---------------------------------
Random.seed!(1)
pre, post = IF(N = 100), IF(N = 400)
param = iSTDPRate(η = 0.2f0, r = 5Hz, τy = 20ms, Wmin = 0.0f0, Wmax = 200.0f0)
syn = SpikingSynapse(pre, post, :gi; conn = (p = 0.2f0, μ = 10.0f0, σ = 2.0f0), LTPParam = param)
W0, I0, J0 = copy(syn.W), copy(syn.I), copy(syn.J)
T = Time(); dt = 0.1f0
Random.seed!(2)
for step = 1:2000
    pre.fire .= rand(pre.N) .< 0.01
    post.fire .= rand(post.N) .< 0.01
    update_time!(T, dt)
    SNNModels.plasticity!(syn, param, syn.LTPVars, dt, T)
end
# plain reference: what the rule should give for this exact spike sequence
Random.seed!(1)
pre2, post2 = IF(N = 100), IF(N = 400)
ref = SpikingSynapse(pre2, post2, :gi; conn = (p = 0.2f0, μ = 10.0f0, σ = 2.0f0), LTPParam = param)
tpre, tpost, Wr = zeros(Float32, 100), zeros(Float32, 400), copy(ref.W)
Random.seed!(2)
for step = 1:2000
    fJ = rand(100) .< 0.01; fI = rand(400) .< 0.01
    for j = 1:100
        tpre[j] += dt * (-tpre[j]) / param.τy
        if fJ[j]
            tpre[j] += 1
            for s = ref.colptr[j]:(ref.colptr[j+1]-1)
                Wr[s] = clamp(Wr[s] + param.η * (tpost[ref.I[s]] - 2 * param.r * param.τy), param.Wmin, param.Wmax)
            end
        end
    end
    for i = 1:400
        tpost[i] += dt * (-tpost[i]) / param.τy
        if fI[i]
            tpost[i] += 1
            for s in eachindex(ref.I)
                ref.I[s] == i && (Wr[s] = clamp(Wr[s] + param.η * tpre[ref.J[s]], param.Wmin, param.Wmax))
            end
        end
    end
end
kernel_err = maximum(abs.(syn.W .- Wr))

# --- 2. network level: E-I network with iSTDP on I->E, train! ---------------------------
Random.seed!(3)
E = IF(N = 800); Ipop = IF(N = 200)
EE = SpikingSynapse(E, E, :ge; conn = (p = 0.1f0, μ = 2.0f0))
EI = SpikingSynapse(E, Ipop, :ge; conn = (p = 0.1f0, μ = 2.0f0))
IE = SpikingSynapse(Ipop, E, :gi; conn = (p = 0.1f0, μ = 10.0f0),
                    LTPParam = iSTDPRate(η = 0.2f0, r = 5Hz, τy = 20ms, Wmin = 0.0f0, Wmax = 200.0f0))
II = SpikingSynapse(Ipop, Ipop, :gi; conn = (p = 0.1f0, μ = 10.0f0))
noiseE = Stimulus(PoissonLayer(8Hz, N = 400), E, :ge; conn = (p = 0.1f0, μ = 3.0f0))
noiseI = Stimulus(PoissonLayer(8Hz, N = 400), Ipop, :ge; conn = (p = 0.1f0, μ = 3.0f0))
model = compose(; E, Ipop, EE, EI, IE, II, noiseE, noiseI, silent = true)
WIE0 = copy(IE.W)
monitor!(E, [:fire])
train!(model = model, duration = 5s, dt = 0.1f0, pbar = false)
st = spiketimes(E)
rates = [count(t -> t > 2000f0, s) / 3.0 for s in st]   # Hz over the last 3 s
@save joinpath(@__DIR__, "result_$(label).jld2") W0 I0 J0 W_kernel = syn.W kernel_err WIE0 WIE = IE.W rates
println(label, ": kernel max |W - plain reference| = ", kernel_err,
        " | network mean E rate = ", round(mean(rates), digits = 3), " Hz, sd = ", round(std(rates), digits = 3),
        " | mean IE weight = ", round(mean(IE.W), digits = 3))
