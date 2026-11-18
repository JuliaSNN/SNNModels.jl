#=
STDPTriplet (Pfister & Gerstner 2006), all-to-all and nearest-spike traces: closed-form updates of one
synapse and equality of the event-driven rule with a dense per-step Float64 reference.
=#

using SNNModels
using Test
using Random
@load_units

const _TDT = 0.1f0     # time step (ms)

# trace value at step n of a neuron that fired at `steps` (all-to-all: sum, nearest: last spike only),
# read before the spikes of step n
function _trace(steps, n, τ, nearest; dt = _TDT)
    past = filter(<(n), steps)
    isempty(past) && return 0.0
    nearest && return exp(-(n - maximum(past)) * dt / τ)
    return sum(exp(-(n - m) * dt / τ) for m in past)
end

# closed-form total weight change of one synapse for given pre and post spike steps
function _triplet_dw(p, pre_steps, post_steps)
    dw = 0.0
    for a in pre_steps         # depression at presynaptic spikes
        dw -= _trace(post_steps, a, p.τ_minus, p.nearest) * (p.A2_minus + p.A3_minus * _trace(pre_steps, a, p.τ_x, p.nearest))
    end
    for b in post_steps        # potentiation at postsynaptic spikes
        dw += _trace(pre_steps, b, p.τ_plus, p.nearest) * (p.A2_plus + p.A3_plus * _trace(post_steps, b, p.τ_y, p.nearest))
    end
    return dw
end

# one synapse between Identity populations, spikes forced at the given steps
function _triplet_run(p, pre_steps, post_steps; nsteps = maximum(vcat(pre_steps, post_steps)) + 5, w0 = 1.0f-3)
    pre, post = Identity(N = 1), Identity(N = 1)
    syn = SpikingSynapse(pre, post, :g; conn = fill(w0, 1, 1), LTPParam = p)   # small w0: dw keeps the Float32 digits
    T = Time()
    for n in 1:nsteps
        pre.fire[1] = n in pre_steps
        post.fire[1] = n in post_steps
        update_time!(T, _TDT)
        plasticity!(syn, syn.param, _TDT, T)
    end
    return Float64(syn.W[1]) - w0
end

# Dense Float64 reference: every trace decays exactly every step; weights read the traces before the
# spikes of the step; then the traces of the spiking neurons are increased by 1 or set to 1.
function _triplet_dense(p, pre_fire, post_fire, I, J, W0)
    W = Float64.(W0)
    Npre, Npost = size(pre_fire, 1), size(post_fire, 1)
    r1, r2, o1, o2 = zeros(Npre), zeros(Npre), zeros(Npost), zeros(Npost)
    f = (r1 = exp(-_TDT / p.τ_plus), r2 = exp(-_TDT / p.τ_x), o1 = exp(-_TDT / p.τ_minus), o2 = exp(-_TDT / p.τ_y))
    for n in 1:size(pre_fire, 2)
        for s in eachindex(W)
            i, j = I[s], J[s]
            pre_fire[j, n] && (W[s] = clamp(W[s] - o1[i] * (p.A2_minus + p.A3_minus * r2[j]), p.Wmin, p.Wmax))
        end
        for s in eachindex(W)
            i, j = I[s], J[s]
            post_fire[i, n] && (W[s] = clamp(W[s] + r1[j] * (p.A2_plus + p.A3_plus * o2[i]), p.Wmin, p.Wmax))
        end
        for (x, fire) in ((r1, view(pre_fire, :, n)), (r2, view(pre_fire, :, n)), (o1, view(post_fire, :, n)), (o2, view(post_fire, :, n)))
            for k in eachindex(x)
                fire[k] && (x[k] = p.nearest ? 1.0 : x[k] + 1.0)
            end
        end
        r1 .*= f.r1; r2 .*= f.r2; o1 .*= f.o1; o2 .*= f.o2
    end
    return W
end

@testset "STDPTriplet: closed-form updates, all-to-all and nearest-spike" begin
    # two presynaptic spikes, three postsynaptic spikes, then two presynaptic spikes:
    # pairs, post-post triplets (o2) and pre-pre triplets (r2) all contribute
    pre_steps, post_steps = [10, 30, 120, 125], [60, 75, 90]
    for nearest in (false, true)
        p = STDPTriplet(A2_plus = 1.0e-2, A3_plus = 2.0e-2, A2_minus = 1.5e-2, A3_minus = 5.0e-3, τ_x = 100.0, τ_y = 60.0,
                        Wmin = -1.0, Wmax = 1.0, nearest = nearest)
        @test _triplet_run(p, pre_steps, post_steps) ≈ _triplet_dw(p, pre_steps, post_steps) rtol = 1.0e-4
    end
    # nearest-spike and all-to-all differ only through the earlier spikes
    kw = (A2_plus = 1.0e-2, A3_plus = 1.0e-2, A2_minus = 1.0e-2, A3_minus = 1.0e-2, Wmin = -1.0, Wmax = 1.0)
    p_all, p_near = STDPTriplet(; kw...), STDPTriplet(; kw..., nearest = true)
    @test _triplet_run(p_all, [10], [20]) ≈ _triplet_run(p_near, [10], [20]) rtol = 1.0e-6
    @test !(_triplet_run(p_all, [5, 10], [20]) ≈ _triplet_run(p_near, [5, 10], [20]))
    @test convert_precision(p_near, Float64).nearest
end

@testset "STDPTriplet: event-driven equals the dense reference" begin
    Npre, Npost, nsteps = 60, 40, 6000
    for nearest in (false, true), full in (true, false)
        # full: four amplitudes; minimal: A2_plus = A3_minus = 0
        p = STDPTriplet(A2_plus = full ? 0.05 : 0.0, A3_plus = 0.1, A2_minus = 0.1, A3_minus = full ? 0.01 : 0.0,
                        τ_x = 100.0, τ_y = 60.0, Wmin = 0.0, Wmax = 2.0, nearest = nearest)
        Random.seed!(11)
        pre, post = Identity(N = Npre), Identity(N = Npost)
        syn = SpikingSynapse(pre, post, :g; conn = (p = 0.2f0, μ = 1.0f0, σ = 0.2f0), LTPParam = p)
        W0 = copy(syn.W)
        pre_fire = rand(Npre, nsteps) .< 0.004
        post_fire = rand(Npost, nsteps) .< 0.004
        T = Time()
        for n in 1:nsteps
            pre.fire .= view(pre_fire, :, n)
            post.fire .= view(post_fire, :, n)
            update_time!(T, _TDT)
            plasticity!(syn, syn.param, _TDT, T)
        end
        R = _triplet_dense(p, pre_fire, post_fire, syn.I, syn.J, W0)
        change = maximum(abs.(R .- W0))
        @test change > 1.0e-2
        @test maximum(abs.(syn.W .- R)) < 1.0e-4 * change
    end
end
