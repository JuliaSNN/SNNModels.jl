# SNNModels 2.0.3: Int32 neuron ids in the :fire COO record, files of 2.0.2 (Int64 ids), clear_monitor! of a
# Time{Float32}, monitor_time sizing of :fire / dense records across chunks, get_git_commit_hash(; dir).
using SNNModels, Test, Random, JLD2
using SNNModels.Logging
@load_units

function _tonic203(; N = 50)
    Random.seed!(11)
    pop = IF(N = N, param = IFParameter(El = -49mV))   # deterministic tonic firing
    return pop, compose(pop = pop, silent = true)
end

@testset "Int32 neuron ids in :fire" begin
    pop, model = _tonic203()
    monitor!(pop, [:fire])
    sim!(model, 200ms)
    rec = pop.records[:fire]
    @test rec[:neurons_buf] isa Vector{Int32}
    n = sum(length, spiketimes(pop))
    @test n > 100
    @test sizeof(eltype(rec[:times_buf])) + sizeof(eltype(rec[:neurons_buf])) == 8   # was 12 with Int64 ids
    # clearing keeps the element type
    clear_records!(pop)
    @test pop.records[:fire][:neurons_buf] isa Vector{Int32}
    # more neurons than Int32 can number
    @test_throws ArgumentError SNNModels._allocate_fire!(pop, pop.records[:meta], Int(typemax(Int32)) + 1, 100f0)
end

@testset "files saved with 2.0.2 (Int64 ids) read and continue identically" begin
    ref, mref = _tonic203()
    monitor!(ref, [:fire])
    sim!(mref, 400ms)

    pop, model = _tonic203()
    monitor!(pop, [:fire])
    sim!(model, 200ms)
    # what 2.0.2 stored: Int64 ids
    pop.records[:fire][:neurons_buf] = Vector{Int}(pop.records[:fire][:neurons_buf])
    @test pop.records[:fire][:neurons_buf] isa Vector{Int}
    dir = mktempdir()
    try
        info = (n = 1,)
        save_model(model = model, path = dir, name = "old202", info = info)
        loaded = load_data(dir, "old202", info).model
        lpop = loaded.pop.pop
        @test lpop.records[:fire][:neurons_buf] isa Vector{Int}      # readers take both element types: no conversion on load
        @test spiketimes(lpop) == spiketimes(pop)
        @test firing_rate(lpop; interval = 0:10:200) == firing_rate(pop; interval = 0:10:200)
        # continuing the loaded model narrows the buffer at the next sim! and gives the single-run result
        sim!(loaded, 200ms)
        @test lpop.records[:fire][:neurons_buf] isa Vector{Int32}
        @test spiketimes(lpop) == spiketimes(ref)
    finally
        rm(dir; recursive = true, force = true)
    end
end

@testset "clear_monitor! of a model: no warning for Time{Float32}" begin
    pop, model = _tonic203()
    monitor!(pop, [:fire])
    @test_logs clear_monitor!(model)   # fails on any log record
    @test isempty(pop.records)
end

@testset "monitor_time sizes :fire and dense records once across chunks" begin
    ref, mref = _tonic203()
    monitor!(ref, [:fire, :v]; sr = 1kHz)
    sim!(mref, 2000ms)

    pop, model = _tonic203()
    monitor!(pop, [:fire, :v]; sr = 1kHz, monitor_rate = 100Hz, monitor_time = 2000ms)
    logger = Test.TestLogger(min_level = Logging.Warn)
    caps = Int[]
    with_logger(logger) do
        for _ = 1:80
            sim!(model, 25ms)
            push!(caps, length(pop.records[:fire][:times_buf]))
        end
    end
    @test isempty(logger.logs)
    @test length(unique(caps)) == 1                       # one allocation, capacity never changes
    @test caps[1] == ceil(Int, 100 * 50 * 2.0 * 1.1)      # rate * N * monitor_time * 1.1
    @test size(pop.records[:v], 2) == 2001                # dense: 2 s at 1 kHz + the t = 0 sample
    @test spiketimes(pop) == spiketimes(ref)
    @test record(pop, :v) == record(ref, :v)

    # an underestimated monitor_time falls back to geometric growth (correct data, one warning)
    pop2, model2 = _tonic203()
    monitor!(pop2, [:fire]; monitor_rate = 1Hz, monitor_time = 100ms)
    logger2 = Test.TestLogger(min_level = Logging.Warn)
    with_logger(logger2) do
        for _ = 1:80
            sim!(model2, 25ms)
        end
    end
    @test spiketimes(pop2) == spiketimes(ref)
    @test count(l -> occursin("Fire COO buffer", string(l.message)), logger2.logs) == 1
end

@testset "get_git_commit_hash(; dir)" begin
    repo = mktempdir()
    plain = mktempdir()
    try
        run(pipeline(`git -C $repo init -q`; stderr = devnull))
        run(pipeline(`git -C $repo -c user.name=t -c user.email=t@t commit -q --allow-empty -m x`; stderr = devnull))
        expected = readchomp(`git -C $repo rev-parse HEAD`)
        withenv("GIT_DIR" => nothing, "GIT_WORK_TREE" => nothing, "GIT_CEILING_DIRECTORIES" => tempdir()) do
            # process started outside any repository
            cd(plain) do
                @test SNNModels.get_git_commit_hash(; dir = repo) == expected
                @test (@test_logs (:warn, r"no git repository") SNNModels.get_git_commit_hash()) == "unknown"
                @test SNNModels.get_git_commit_hash(; warn = false) == "unknown"
                @test SNNModels.get_git_commit_hash(; dir = joinpath(plain, "missing"), warn = false) == "unknown"
                # write_config records the repository given by commit_dir
                f = SNNModels.write_config(joinpath(plain, "config.jl"), (a = 1,); config = nothing, commit_dir = repo)
                @test occursin(expected, read(f, String))
            end
        end
    finally
        rm(repo; recursive = true, force = true)
        rm(plain; recursive = true, force = true)
    end
end
