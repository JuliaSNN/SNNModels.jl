# Benchmark of disconnected (length-0) dendrites in Multipod (not part of the test suite).
#
# Usage (any env that develops SNNModels):
#   julia --project=<env> -t 1 bench/multipod_inactive_bench.jl
#
# Question: does a per-neuron mask that skips the work of a disconnected dendrite save time? Float32, N = 2000, dt = 0.1 ms,
# integrate! alone (2000 steps), Poisson events (20 Hz x 100 sources, ρ = 0.5) on the soma and on every connected dendrite.
#   A. the library: Multipod with 6 dendrites of which 0, 2 or 4 are disconnected in all neurons (length 0), and in half of the
#      neurons, against Multipod with 4 and 2 dendrites;
#   B. a prototype of the mask: the two generated per-neuron loops are redefined here (this process only) so that the block of a
#      dendrite of length <= 0 is skipped for that neuron (frozen dendrite, no axial current), then the same runs.
# Kept in bench/ to document why the library has no such mask.

using SNNModels
using Random
using Printf
using SNNModels.UnPack: @unpack
@load_units

const N = 2000
const DT = 0.1f0
const STEPS = 2000
const LENGTHS = [400um, 150um, 300um, 250um, 200um, 350um]

# a Multipod with the first `D` lengths; the dendrites `off` disconnected in the neurons `who`
function population(D; off = Int[], who = 1:N)
    Random.seed!(1)
    M = Multipod(N = N, param = MultipodParameter(ds = LENGTHS[1:D]))
    for k in off
        set_dendrite_length!(M, k, 0um; neurons = who)
    end
    return M
end

# pre-drawn Poisson events on the soma and on the connected dendrites `active`
function events(active, steps; seed = 3)
    rng = Xoshiro(seed)
    p = 100 * 20e-3 * DT * 0.5
    return [findall(rand(rng, N) .< p) for _ in 1:(length(active) + 1), _ in 1:steps]
end

function run!(M, active, ev, steps)
    bufs = (M.receptors_s.glu, (getfield(M, Symbol(:receptors_d, k)).glu for k in active)...)
    for tt in 1:steps
        for (c, buf) in enumerate(bufs)
            @inbounds for i in ev[c, tt]
                buf[i] += 4f0
            end
        end
        SNNModels.integrate!(M, M.param, DT)
    end
end

function timing(D; off = Int[], who = 1:N, reps = 5)
    M = population(D; off, who)
    active = setdiff(1:D, who == 1:N ? off : Int[])
    ev = events(active, STEPS)
    run!(M, active, ev, 50)
    best = minimum(@elapsed(run!(M, active, ev, STEPS)) for _ in 1:reps)
    return best / (STEPS * N) * 1e9
end

const CASES = [
    ("Multipod D=6, all connected", 6, Int[], 1:N),
    ("Multipod D=6, 2 off (all neurons)", 6, [5, 6], 1:N),
    ("Multipod D=6, 4 off (all neurons)", 6, [3, 4, 5, 6], 1:N),
    ("Multipod D=6, 4 off (half the neurons)", 6, [3, 4, 5, 6], 1:2:N),
    ("Multipod D=4", 4, Int[], 1:N),
    ("Multipod D=2", 2, Int[], 1:N),
]

function measure(tag)
    out = Dict{String,Float64}()
    for (name, D, off, who) in CASES
        out[name] = timing(D; off, who)
    end
    return out
end

library = measure("library")

# ---------------------------------------------------------------------------------------------------------------------------------
# B. Prototype of the per-neuron mask: the loops of `multipod.jl` with each dendrite block under `if l > 0`
# ---------------------------------------------------------------------------------------------------------------------------------

@eval SNNModels @generated function _multipod_derivatives!(p::Multipod{D}, Δv::Matrix{FT}, dt::FT, rng::UnitRange{Int}) where {D,FT<:AbstractFloat}
    vd = [Symbol(:vd, k) for k in 1:D]
    ic = [Symbol(:ic, k) for k in 1:D]
    blocks = [quote
        if len[$k][i] > 0
            $(vd[k]) = v_pred[i, $(k + 1)]
            $(ic[k]) = (vs - $(vd[k])) * gax[$k][i]
            Δv[i, $(k + 1)] = ((muladd(Eld[$k][i] - $(vd[k]), gm[$k][i], -is[i, $(k + 1)]) + $(ic[k])) + I_d[i]) / Cd[$k][i]
        else
            $(ic[k]) = zero(FT)
            Δv[i, $(k + 1)] = zero(FT)
        end
    end for k in 1:D]
    outgoing = foldl((acc, k) -> :($acc + $(ic[k])), 1:D; init = :(ws + is[i, 1]))
    cable(f) = Expr(:tuple, (:(dends[$k].$f) for k in 1:D)...)
    return quote
        @unpack v_s, I_d, I, w_s, θ, is, v_pred, adex = p
        @unpack C, gl, El, ΔT, a, τw = adex
        dends = _dendrites(p)
        gax = $(cable(:gax)); gm = $(cable(:gm)); Eld = $(cable(:El)); Cd = $(cable(:C)); len = $(cable(:l))
        @inbounds @simd ivdep for i ∈ rng
            vs = v_pred[i, 1]
            ws = muladd(Δv[i, $(D + 2)], dt, w_s[i])
            $(blocks...)
            e = exp256((vs - θ[i]) / ΔT)
            Δv[i, 1] = muladd(gl, muladd(ΔT, e, El - vs), I[i] - $outgoing) / C
            Δv[i, $(D + 2)] = muladd(a, vs - El, -ws) / τw
        end
    end
end

masked = measure("mask")

println("Disconnected dendrites: N = $N, Float32, integrate! alone, $(Threads.nthreads()) thread(s); ns per neuron-step")
@printf("   %-42s %10s %16s %8s\n", "population", "library", "mask prototype", "mask/lib")
for (name, _, _, _) in CASES
    @printf("   %-42s %10.2f %16.2f %8.3f\n", name, library[name], masked[name], masked[name] / library[name])
end
