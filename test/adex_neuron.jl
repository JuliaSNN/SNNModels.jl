adex_types = [AdExParameter, AdExSinExpParameter, AdExReceptorParameter]

for adex_type in adex_types
    # @show adex_type
    E = AdEx(; N = 1, param = adex_type())
    E.I .= [11]
    monitor!(E, [:v, :fire])
    sim!([E]; duration = 300ms)
end
