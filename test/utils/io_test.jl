using SNNModels
using Test
using JLD2
import Dates: DateTime
@load_units

@testset "Utils - io.jl" begin
    test_dir = mktempdir()
    try
        @testset "SNNfolder" begin
            info   = (param1 = 1.0, param2 = "test")
            folder = SNNfolder(test_dir, "mymodel", info)
            @test occursin("mymodel", folder)
            @test occursin(test_dir, folder)
        end

        @testset "SNNpath" begin
            info = (N = 100,)
            path = SNNpath(test_dir, "test", info, :model, 0)
            @test endswith(path, ".jld2")
            @test occursin("test", path)
        end

        @testset "Save and load model" begin
            E     = IF(N = 10)
            model = compose(E = E, silent = true)
            info  = (N = 10, test = true)
            saved_path = save_model(
                model = model,
                path  = test_dir,
                name  = "test_model",
                info  = info,
            )
            @test isfile(saved_path)
            loaded = load_model(test_dir, "test_model", info)
            @test loaded.model.pop.E.N == 10
        end

        @testset "get_timestamp" begin
            ts = SNNModels.get_timestamp()
            @test ts isa DateTime
        end

    finally
        rm(test_dir; recursive = true, force = true)
    end
end

@testset "get_git_commit_hash outside a git repository" begin
    # cluster jobs often run from a copied directory with no .git: must not throw
    h = cd(mktempdir()) do
        withenv("GIT_DIR" => nothing, "GIT_WORK_TREE" => nothing, "GIT_CEILING_DIRECTORIES" => tempdir()) do
            SNNModels.get_git_commit_hash()
        end
    end
    @test h == "unknown"
    # inside this repository it still returns a 40-character hash
    @test length(cd(SNNModels.get_git_commit_hash, pkgdir(SNNModels))) == 40
end
