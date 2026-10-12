using SNNModels
using Test
using Random
using SNNModels.Distributions
@load_units

# The precision of a model is chosen once, on the populations (`VFT = Vector{Float64}`); connections, stimuli, the state of
# the plasticity rules, delay buffers and the clock follow (`float_type`). The same small network is built in both
# precisions, with the same rule, and the types of all state arrays are checked.

# fields that are Float64 in every precision by design (accumulators, scales, caches, trace amplitudes of the event vSTDP)
const _F64_FIELDS = (:scale, :decay, :K, :snap, :pw, :cache, :x_amplitude)

function _float_fields(obj; skip = ())
    out = Pair{Symbol,DataType}[]
    for f in fieldnames(typeof(obj))
        f in skip && continue
        v = getfield(obj, f)
        v isa AbstractArray{<:AbstractFloat} && push!(out, f => eltype(v))
        v isa AbstractFloat && push!(out, f => typeof(v))
    end
    return out
end

function _precision_net(FT; rule = STDPGerstner(FT = FT), stp = NoSTP(), delay = nothing, Npre = 8, Npost = 5)
    V = Vector{FT}
    pre = Identity(N = Npre, VFT = V)
    pre2 = Identity(N = Npre, VFT = V)
    post = IF(N = Npost, VFT = V)
    stimP = PoissonStimulus(pre, :g; param = PoissonFixed(rate = 0.05))
    stimT = SpikeTimeStimulusIdentity(pre2, :g; param = SpikeTimeParameter([5.0, 7.0, 20.0, 40.0], [1, 2, 3, 4]))
    stimD = PoissonStimulus(post, :ge; param = PoissonFixed(rate = 0.1, μ = 20.0))
    syn = SpikingSynapse(pre, post, :ge; conn = (p = 0.8, μ = 1.0), LTPParam = rule, STPParam = stp, delay_dist = delay)
    syn2 = SpikingSynapse(pre2, post, :ge; conn = (p = 0.8, μ = 1.0))
    return (; pre, pre2, post, stimP, stimT, stimD, syn, syn2)
end

# rules of the test, as functions of the precision (the rule is built with its own FT, as a user would)
const _PRECISION_RULES = (
    "STDPGerstner pair" => FT -> STDPGerstner(FT = FT, A_pre = 1e-2, A_post = -1.2e-2),
    "STDPTriplet" => FT -> STDPTriplet(FT = FT, A2_plus = 1e-2, A3_plus = 1e-2, A2_minus = 1e-2, A3_minus = 1e-3),
    "vSTDPParameter (event)" => FT -> vSTDPParameter(FT = FT, A_LTP = 1e-2, Wmax = 5.0, Wmin = 0.0),
    "vSTDPDenseParameter" => FT -> vSTDPDenseParameter(FT = FT, A_LTP = 1e-2, Wmax = 5.0, Wmin = 0.0),
)

@testset "Float32 and Float64 from the population type" begin
    @testset "float_type" begin
        @test float_type(Identity(N = 3)) == Float32
        @test float_type(Identity(N = 3, VFT = Vector{Float64})) == Float64
        @test float_type(IF(N = 3)) == Float32
        @test float_type(IF(N = 3, VFT = Vector{Float64})) == Float64
        @test float_type(Time()) == Float32 && float_type(Time{Float64}()) == Float64
        @test float_type(Float64) == Float64 && float_type(Vector{Float32}) == Float32
        @test float_type(NoLTP()) == Float32
        @test float_type(STDPGerstner(FT = Float64)) == Float64
    end

    @testset "population defaults follow VFT" begin
        P = IF(N = 4, VFT = Vector{Float64})
        @test all(a -> eltype(a) == Float64, (P.v, P.w, P.I, P.syn_curr, P.receptors.glu, P.receptors.gaba,
                                               P.synvars.ge, P.synvars.gi, P.synvars.he, P.synvars.hi))
        @test P.param isa IFParameter{Float64} && P.synapse isa DoubleExpSynapse{Float64} && P.spike isa PostSpike{Float64}
        P32 = IF(N = 4)
        @test all(a -> eltype(a) == Float32, (P32.v, P32.w, P32.I, P32.syn_curr, P32.receptors.glu, P32.synvars.ge))
        I64 = Identity(N = 4, VFT = Vector{Float64})
        @test all(a -> eltype(a) == Float64, (I64.g, I64.h, I64.spikecount))
        # a population that mixes precisions is refused
        @test_throws ArgumentError IF(N = 3, VFT = Vector{Float64}, param = IFParameter())
        @test_throws ArgumentError IF(N = 3, synapse = DoubleExpSynapse(FT = Float64))
    end

    @testset "$name, $FT" for (name, mkrule) in _PRECISION_RULES, FT in (Float32, Float64)
        Random.seed!(7)
        n = _precision_net(FT; rule = mkrule(FT))
        (; pre, pre2, post, stimP, stimT, stimD, syn, syn2) = n
        # everything the connection and the stimuli own has the precision of the populations
        for obj in (syn, syn2)
            @test float_type(obj) == FT
            @test eltype(obj.W) == FT && eltype(obj.ρ) == FT && eltype(obj.g) == FT && eltype(obj.v_post) == FT
        end
        for st in (stimP, stimD)
            @test float_type(st) == FT && eltype(st.g) == FT && float_type(st.param) == FT
        end
        @test float_type(stimT) == FT && eltype(stimT.W) == FT && eltype(stimT.g) == FT
        @test eltype(stimT.param.spiketimes) == FT && eltype(stimT.next_spike) == FT
        @test float_type(syn.LTPParam) == FT
        skip = syn.LTPVars isa vSTDPVariables ? (_F64_FIELDS..., :x) : _F64_FIELDS
        @test all(p -> p.second == FT, _float_fields(syn.LTPVars; skip))
        @test all(p -> p.second == Float64, _float_fields(syn.LTPVars; skip = setdiff(fieldnames(typeof(syn.LTPVars)), _F64_FIELDS)))
        # a Float64 clock for a Float64 model, a Float32 one otherwise
        model = compose(; n..., silent = true)
        @test typeof(model.time) == Time{FT}
        # a step of every kind of component is type stable (one precision, no promotion in the step)
        T = model.time
        dt = FT(0.1)
        update_time!(T, dt)
        @test (@inferred SNNModels.stimulate!(stimP, stimP.param, T, dt)) === nothing
        @test (@inferred SNNModels.stimulate!(stimT, stimT.param, T, dt)) === nothing
        @test (@inferred SNNModels.integrate!(pre, pre.param, dt)) === nothing
        @test (@inferred SNNModels.integrate!(post, post.param, dt)) === nothing
        SNNModels.prepare_spikes!(syn)
        @test (@inferred SNNModels.update_traces!(syn, syn.param, dt, T)) === nothing
        @test (@inferred SNNModels.forward!(syn, syn.param, dt, T)) === nothing
        @test (@inferred SNNModels.plasticity!(syn, syn.param, dt, T)) === nothing
        SNNModels.release_spikes!(syn)
        # a run keeps every array in its precision and the weights finite and in bounds
        train!(model, 300ms; dt = dt)
        @test typeof(model.time) == Time{FT} && eltype(syn.W) == FT && eltype(post.v) == FT
        W = plasticity_value(syn, :W)
        @test eltype(W) == FT && all(isfinite, W) && all(>=(0), W)
        @test any(!=(1), W)                       # plasticity acted
    end

    @testset "STP event, delays, Poisson layer, $FT" for FT in (Float32, Float64)
        n = _precision_net(FT; stp = MarkramSTPParameter(FT = FT), delay = Uniform(0.5, 3.0))
        (; pre, post, syn) = n
        @test eltype(syn.ρ) == FT && eltype(syn.STPVars.u) == FT && eltype(syn.STPVars.x) == FT && eltype(syn.STPVars._ρ) == FT
        @test eltype(syn.STPVars.last_spike) == FT
        @test eltype(syn.param.delaytime) == FT && eltype(syn.param.buffer) == FT && eltype(syn.param.dt) == FT
        @test float_type(syn.STPParam) == FT
        model = compose(; n..., silent = true)
        train!(model, 200ms; dt = FT(0.1))
        @test eltype(syn.param.buffer) == FT && eltype(syn.STPVars.u) == FT
        @test eltype(plasticity_value(syn, :STPVars, :u, 200)) == FT
        T = model.time; dt = FT(0.1)
        update_time!(T, dt)
        @test (@inferred SNNModels.update_traces!(syn, syn.param, dt, T)) === nothing
        @test (@inferred(SNNModels.forward!(syn, syn.param, dt, T)); true)   # the delay method returns the ring position
        # Poisson layer onto the population
        layer = Stimulus(PoissonLayer(rate = 0.05, N = 6), post, :ge; conn = (p = 0.5, μ = 1.0))
        @test float_type(layer) == FT && eltype(layer.W) == FT && eltype(layer.g) == FT && float_type(layer.param) == FT
        @test (@inferred SNNModels.stimulate!(layer, layer.param, T, dt)) === nothing
    end

    @testset "metaplasticity follows the synapses, $FT" for FT in (Float32, Float64)
        n = _precision_net(FT)
        norm = SynapseNormalization([n.syn]; param = MultiplicativeNorm(τ = 5ms))
        @test eltype(norm.W0) == FT && eltype(norm.μ) == FT && float_type(norm.param) == FT
        agg = AggregateScaling(n.post, [n.syn]; param = AggregateScalingParameter(n.post.N, 5Hz))
        @test eltype(agg.WT) == FT && eltype(agg.y) == FT && float_type(agg.param) == FT
        model = compose(; n..., norm, agg, silent = true)
        train!(model, 50ms; dt = FT(0.1))
        @test eltype(n.syn.W) == FT
    end

    @testset "Float64 values are Float64 (analytic pair STDP)" begin
        # one pre spike, then a post spike 40 steps later: dw = A_pre exp(-Δt/τ), to the rounding of the precision of the
        # synapse (a Float32 intermediate would show as a relative error of 1e-8 in the Float64 run)
        for FT in (Float32, Float64)
            pre, post = Identity(N = 1, VFT = Vector{FT}), Identity(N = 1, VFT = Vector{FT})
            rule = STDPGerstner(FT = FT, A_pre = 1e-3, A_post = 0.0, τpre = 20.0, τpost = 20.0, Wmin = 0.0, Wmax = 10.0)
            syn = SpikingSynapse(pre, post, :g; conn = fill(0.01, 1, 1), LTPParam = rule)
            T = Time{FT}(); dt = FT(0.1)
            for n in 1:41
                update_time!(T, dt)
                pre.fire[1] = n == 1
                post.fire[1] = n == 41
                SNNModels.plasticity!(syn, syn.param, dt, T)
            end
            expected = 1e-3 * exp(-Float64(dt) * 40 / 20)
            err = abs((Float64(syn.W[1]) - Float64(FT(0.01))) - expected) / expected
            @test err < (FT == Float64 ? 1e-12 : 1e-3)
        end
    end

    @testset "mixed precision" begin
        # a Float32 presynaptic population onto a Float64 one: the synapse has the precision of its target
        pre, post = Identity(N = 4), Identity(N = 3, VFT = Vector{Float64})
        syn = SpikingSynapse(pre, post, :g; conn = (p = 1.0, μ = 1.0), LTPParam = STDPGerstner())
        @test float_type(syn) == Float64 && eltype(syn.W) == Float64 && float_type(syn.LTPParam) == Float64
        # the reverse: a Float64 rule on a Float32 population is converted to Float32
        syn32 = SpikingSynapse(Identity(N = 4), Identity(N = 3), :g; conn = (p = 1.0, μ = 1.0), LTPParam = STDPGerstner(FT = Float64))
        @test float_type(syn32) == Float32 && float_type(syn32.LTPParam) == Float32 && eltype(syn32.LTPVars.tpre) == Float32
        # a model of populations of both precisions runs, each component in its own precision; the clock is Float64
        a, b = Identity(N = 4, VFT = Vector{Float64}), Identity(N = 4)
        s1 = SpikingSynapse(a, b, :g; conn = (p = 1.0, μ = 1.0))
        s2 = SpikingSynapse(b, a, :g; conn = (p = 1.0, μ = 1.0))
        model = compose(; a, b, s1, s2, silent = true)
        @test typeof(model.time) == Time{Float64} && eltype(s1.W) == Float32 && eltype(s2.W) == Float64
        a.g .= 1.0
        sim!(model, 5ms)
        @test eltype(a.g) == Float64 && eltype(b.g) == Float32
        # components that have no Float64 model refuse a Float64 target at construction (never a silent copy)
        P64 = IF(N = 3, VFT = Vector{Float64})
        @test_throws ArgumentError CurrentStimulus(P64, :I; param = CurrentNoise(3))
        @test_throws ArgumentError BalancedStimulus(P64, :ge, :gi; param = 5.0)
        @test_throws ArgumentError MetaPlasticity(RandomTurnover(rate = 1e-3), SpikingSynapse(a, P64, :ge; conn = (p = 1.0, μ = 1.0)))
        # populations without a Float64 model refuse a Float64 step (since 1.10 every population model but HetRec has one, see
        # test/pop/precision_populations.jl)
        H = SNNModels.Population(SNNModels.HetRecParameter(); N = 3)
        @test_throws ArgumentError SNNModels.integrate!(H, H.param, 0.1)
    end

    @testset "Float32 default unchanged" begin
        n = _precision_net(Float32; rule = STDPGerstner())
        model = compose(; n..., silent = true)
        @test typeof(model.time) == Time{Float32}
        @test eltype(n.syn.W) == Float32 && float_type(n.syn.LTPParam) == Float32
        # the rule of a Float32 synapse is the object that was passed (no copy)
        rule = vSTDPParameter()
        s = SpikingSynapse(Identity(N = 3), Identity(N = 3), :g; conn = (p = 1.0, μ = 1.0), LTPParam = rule)
        @test s.LTPParam === rule
    end
end
true
