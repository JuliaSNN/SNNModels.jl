# Effect of the membrane pair rule (SNNModels 1.9.0) on Tripod and AdEx when τm is changed with
# @update!, as SNN2AC_optim does (`network.exc.adex.τm = τm`, τm optimised in [5, 20] ms).
#
# Up to 1.8.x `@update!` changed only the τm field. AdEx integrates with τm (and R), so it
# responded; Tripod integrates with C and gl, which stayed at 281 pF / 40 nS, so every τm gave the
# same neuron. Here the "1.8.x" Tripod is therefore AdExParameter(C = 281pF, gl = 40nS) for every
# τm (what that code integrated), and the "1.9.0" Tripod is AdExParameter() after `@update! τm`
# (C = τm gl). Run with SNNModels >= 1.9.0 (branch fix/sweep-bugs).
#
# Usage:  julia --project=<env> membrane_pair.jl      Output: data/membrane_pair.jld2
using SNNModels, JLD2
@load_units

const OUT = joinpath(@__DIR__, "data", "membrane_pair.jld2")
const τms = Float32[5, 7.025, 10, 15, 20]
const Igrid = collect(Float32, 0:25:3000)
const TSIM, DT = 1000.0f0, 0.1f0

function rates(p, soma_current)
    p.v_s .= p.adex.El; p.v_d1 .= p.adex.El; p.v_d2 .= p.adex.El
    p.w_s .= 0; p.θ .= p.adex.Vt; p.tabs .= 0; p.fire .= false
    soma_current .= Igrid
    n = zeros(Float32, length(Igrid))
    for _ = 1:round(Int, TSIM / DT)
        SNNModels.integrate!(p, p.param, DT)
        n .+= p.fire
    end
    return n ./ (TSIM / 1000)
end

function adex_rates(param)
    p = AdEx(N = length(Igrid), param = param)
    p.v .= param.El; p.I .= Igrid
    n = zeros(Float32, length(Igrid))
    for _ = 1:round(Int, TSIM / DT)
        SNNModels.integrate!(p, p.param, DT)
        n .+= p.fire
    end
    return n ./ (TSIM / 1000)
end

function updated(τm)
    a = AdExParameter()
    @update! a τm = τm
    return a
end

res = Dict{String,Any}("I" => Igrid, "τm" => τms)
res["tripod_old"] = Dict(τm => (p = Tripod(N = length(Igrid), adex = AdExParameter(C = 281pF, gl = 40nS)); rates(p, p.I)) for τm in τms)
res["tripod_new"] = Dict(τm => (p = Tripod(N = length(Igrid), adex = updated(τm)); rates(p, p.I)) for τm in τms)
res["C_new"] = Dict(τm => updated(τm).C for τm in τms)
res["adex"] = Dict(τm => adex_rates(updated(τm)) for τm in τms)
jldsave(OUT; res)
for τm in τms
    println("τm = $τm ms: C = $(res["C_new"][τm]) pF; Tripod rate at 1500 pA old ",
            res["tripod_old"][τm][61], " Hz, new ", res["tripod_new"][τm][61], " Hz; AdEx ", res["adex"][τm][61], " Hz")
end
