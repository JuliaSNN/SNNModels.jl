# Helpers shared by the bit-identity tests of the vectorised neuron loops (`*_update_loop.jl`).
# A converted loop must reproduce the reference loop (the one of SNNModels <= 1.9, kept in the test) to the last bit.

# bit-for-bit equality of two numbers (NaN equals NaN, +0 differs from -0)
_bit_equal(a::AbstractFloat, b::AbstractFloat) = (isnan(a) && isnan(b)) || (typeof(a) == typeof(b) && reinterpret(Unsigned, a) == reinterpret(Unsigned, b))
_bit_equal(a, b) = a == b
loop_same(x::AbstractArray, y::AbstractArray) = size(x) == size(y) && all(i -> _bit_equal(x[i], y[i]), eachindex(x))
loop_same(x, y) = _bit_equal(x, y)

# the listed fields of two populations are bit-identical
loop_state_equal(p, q, fields) = all(f -> loop_same(getfield(p, f), getfield(q, f)), fields)

# first differing field (for the failure message), or nothing
function loop_first_diff(p, q, fields)
    for f in fields
        loop_same(getfield(p, f), getfield(q, f)) || return f
    end
    return nothing
end
