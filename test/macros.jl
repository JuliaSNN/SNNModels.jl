# Define test structs
using Test
@kwdef struct TestStruct
    z::Any
    s::Any
end

@kwdef struct AnotherStruct
    x::Any
    y::Any
end
# Test cases for @update and @update! macros with nested structs
# function test_update_macros_with_structs()


@testset "Update macros" begin
    # Test case 1: Simple struct update
    struct_config1 = TestStruct(10, AnotherStruct(5, 6))
    config1 = (a = 10, b = 2, c = struct_config1)
    @update! config1 begin
        z = 20
        s = 30
    end
    @test config1.a == 10
    @test config1.b == 2
    @test config1.c.z == 10
    @test config1.c.s == AnotherStruct(5, 6)

    # Test case 2: Nested struct update
    struct_config2 = TestStruct(10, AnotherStruct(5, 6))
    config2 = (a = 10, b = 2, c = struct_config2)
    @update! config2 begin
        c.z = 20
        c.s = 30
    end
    @test config2.a == 10
    @test config2.b == 2
    @test config2.c.z == 20
    @test config2.c.s == 30

    # Test case 3: Deep nested struct update
    struct_config3 = TestStruct(10, AdExParameter())
    config3 = (a = 10, b = 2, c = struct_config3)
    @update! config3 begin
        c.s.Vr = 200
    end
    @test config3.a == 10
    @test config3.b == 2
    @test config3.c.z == 10
    @test config3.c.s.Vr == 200

    # Test case 4: Mixed named tuple and struct update
    struct_config4 = TestStruct(10, AnotherStruct(5, 6))
    config4 = (a = 10, b = (d = 2, e = 3), c = struct_config4)
    @update! config4 begin
        b.d = 4
        c.z = 20
        c.s.x = 100
    end
    @test config4.a == 10
    @test config4.b.d == 4
    @test config4.b.e == 3
    @test config4.c.z == 20
    @test config4.c.s.x == 100
    @test config4.c.s.y == 6

    # Test case 5: Create new nested struct
    struct_config5 = TestStruct(10, AnotherStruct(5, 6))
    config5 = (a = 10, b = 2, c = struct_config5)
    @test_throws MethodError @update! config5 begin
        c.s.new_field = 100
    end
    @test config5.a == 10
    @test config5.b == 2
    @test config5.c.z == 10
    @test config5.c.s.x == 5
    @test config5.c.s.y == 6


    config1 = (a = 1, b = (c = 2, d = 3))
    updated1 = @update! config1 begin
        a = 4
    end
    @test updated1.a == 4
    @test updated1.b.c == 2
    @test updated1.b.d == 3

    # Test case 2: Nested update
    config2 = (a = 1, b = (c = 2, d = 3))
    updated2 = @update config2 begin
        b.c = 5
    end
    @test updated2.a == 1
    @test updated2.b.c == 5
    @test updated2.b.d == 3

    # Test case 3: Multiple updates in a block
    config3 = (a = 1, b = (c = 2, d = 3))
    updated3 = @update config3 begin
        a = 4
        b.c = 5
    end
    @test updated3.a == 4
    @test updated3.b.c == 5
    @test updated3.b.d == 3

    # Test case 4: Create new nested structure
    config4 = (a = 1,)
    updated4 = @update config4 begin
        b.c = 2
    end
    @test updated4.a == 1
    @test updated4.b.c == 2

    # Test case 5: Overwrite non-NamedTuple field
    config5 = (a = 1, b = 2)
    @test_throws TypeError updated5 = @update config5 begin
        b.c = 3
    end

    # Test case 6: Simple update with !
    config6 = (a = 1, b = (c = 2, d = 3))
    @update! config6 begin
        a = 4
    end
    @test config6.a == 4
    @test config6.b.c == 2
    @test config6.b.d == 3

    # Test case 7: Nested update with !
    config7 = (a = 1, b = (c = 2, d = 3))
    @update! config7 begin
        b.c = 5
    end
    @test config7.a == 1
    @test config7.b.c == 5
    @test config7.b.d == 3

    # Test case 8: Multiple updates in a block with !
    config8 = (a = 1, b = (c = 2, d = 3))
    @update! config8 begin
        a = 4
        b.c = 5
    end
    @test config8.a == 4
    @test config8.b.c == 5
    @test config8.b.d == 3

    # Test case 9: Create new nested structure with !
    config9 = (a = 1,)
    @update! config9 begin
        b.c = 2
    end
    @test config9.a == 1
    @test config9.b.c == 2

    # Test case 10: Overwrite non-NamedTuple field with !
    config10 = (a = 1, b = 2)
    @test_throws TypeError @update! config10 begin
        b.c = 3
    end
    @test config10.a == 1
    @test config10.b == 2
    # @test config10.b.c == 3

end
