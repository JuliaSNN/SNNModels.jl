if_types = [IFParameter, IFCurrentParameter, IFCurrentDeltaParameter, IFSinExpParameter]

for if_type in if_types
    @show if_type
    E = IF(; N = 1, param = if_type())
    E.I = [11]
    monitor!(E, [:v, :fire])
    sim!([E]; duration = 300ms)
end
