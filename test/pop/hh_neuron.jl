E = HH(; N = 1)
E.I = [0.001]

monitor!(E, [:v])
sim!([E]; dt = 0.01f0, duration = 100ms)
true