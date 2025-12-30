@eval SNNModels begin
        @snn_kw struct StimulusGroup{ST=Vector{AbstractStimulus}, } <: AbstractStimulus
            id::String = randstring(12)
            name::String = "StimulusGroup"
            param::PoissonStimulusParameter
            stimuli::ST
            targets::Dict = Dict()
            records::Dict = Dict()
        end

        function StimulusGroup(param::P, 
                                post::T,  
                                sym::Symbol, 
                                comps::Vector{Symbol};
                                name = "StimulusGroup",
                                kwargs...
                                ) where {T<: AbstractPopulation, P<:AbstractStimulusParameter}
            stimuli = Vector{AbstractStimulus}()
            for comp in comps
                push!(stimuli, Stimulus(param, post, sym; comp=comp, name, kwargs...))
            end
            targets = Dict(:pre => :StimulusGroup, :post => post.id)
            StimulusGroup(;name, param, stimuli, targets)
        end

        function stimulate!(
            p::StimulusGroup,
            param::PoissonStimulusParameter,
            time::Time,
            dt::Float32,
        )
            for stim in p.stimuli
                stimulate!(stim, param, time, dt)
            end
        end

    export  StimulusGroup
end

