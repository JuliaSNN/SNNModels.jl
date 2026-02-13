using SNNModels
using Test
using JLD2
@load_units

@testset "Utils - io.jl" begin
    # Create temporary directory for tests
    test_dir = mktempdir()
    
    try
        @testset "SNNfolder" begin
            info = (param1=1.0, param2="test")
            folder = SNNfolder(test_dir, "mymodel", info)
            @test occursin("mymodel", folder)
            @test occursin(test_dir, folder)
        end

        @testset "SNNfile" begin
            @test SNNfile(:model, 0) == "model.jld2"
            @test SNNfile(:data, 1) == "data-1.jld2"
            @test SNNfile(:model, 5) == "model-5.jld2"
        end

        @testset "SNNpath" begin
            info = (N=100,)
            path = SNNpath(test_dir, "test", info, :model, 0)
            @test endswith(path, "model.jld2")
            @test occursin("test", path)
        end

        @testset "Save and load model" begin
            # Create a simple model
            E = IF(N=10, name=:E)
            model = compose(E=E, name="test_save", silent=true)
            info = (N=10, test=true)
            
            # Save model
            saved_path = save_model(
                model=model,
                path=test_dir,
                name="test_model",
                info=info
            )
            
            @test isfile(saved_path)
            
            # Load model back
            loaded = load_model(test_dir, "test_model", info)
            @test loaded.model.pop.E.N == 10
            @test loaded.model.name == "test_save"
        end

        @testset "get_timestamp and get_git_commit_hash" begin
            ts = SNNModels.get_timestamp()
            @test ts isa DateTime
            
            # Git hash test might fail if not in a git repo
            try
                hash = SNNModels.get_git_commit_hash()
                @test length(hash) == 40  # SHA-1 hash length
            catch
                @warn "Skipping git hash test (not in git repo)"
            end
        end

        @testset "read_folder" begin
            # Create some test files
            E = IF(N=10, name=:E)
            model = compose(E=E, name="test", silent=true)
            
            for i in 1:3
                info = (run=i,)
                save_model(
                    model=model,
                    path=test_dir,
                    name="model$i",
                    info=info
                )
            end
            
            # Read all model files
            files = read_folder(test_dir, type=:model)
            @test length(files) >= 3
            @test all(f -> endswith(f, ".jld2"), files)
        end

    finally
        # Cleanup
        rm(test_dir, recursive=true, force=true)
    end
end
