function synaptic_target(
    targets::Dict,
    post::T,
    sym::Symbol,
    target,
) where {T<:AbstractPopulation}
    @warn "Synaptic target not defined for this type. Please implement a method for $(T)"
    g = zeros(Float32, post.N)
    v_post = zeros(Float32, post.N)
    if isnothing(target)
        g = getfield(post, sym)
        _v = :v
        hasfield(typeof(post), _v) && (v_post = getfield(post, _v))
        push!(targets, :sym => sym)
    elseif typeof(target) == Symbol
        _sym = Symbol("$(sym)_$target")
        _v = Symbol("v_$target")
        g = getfield(post, _sym)
        hasfield(typeof(post), _v) && (v_post = getfield(post, _v))
        push!(targets, :sym => _sym)
    elseif typeof(target) == Int
        if typeof(post) <: AbstractDendriteIF
            _sym = Symbol("$(sym)_d")
            _v = Symbol("v_d")
            g = getfield(post, _sym)[target]
            v_post = getfield(post, _v)[target]
            push!(targets, :sym => Symbol(string(_sym, target)))
        elseif isa(post, AdExMultiTimescale)
            g = getfield(post, sym)[target]
            v_post = getfield(post, :v)
            push!(targets, :sym => Symbol(string(sym, target)))
        end
    end
    return g, v_post
    # return zeros(Float32, post.N), zeros(Float32, post.N)
end

function synaptic_target(
    targets::Dict,
    post::T,
    sym::Nothing,
    target,
) where {T<:AbstractPopulation}
    @warn "Synaptic target not instatiated, returning non-pointing arrays"
    g = zeros(Float32, post.N)
    v = zeros(Float32, post.N)
    return g, v
end
