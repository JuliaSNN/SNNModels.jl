E = IF(; N = 1)
E.I = [11]
monitor!(E, [:v, :fire])

sim!([E]; duration = 300ms)
