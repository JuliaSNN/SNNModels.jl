G = Rate(; N = 100)
GG = RateSynapse(G, G; μ = 1.2, p = 1.0)
monitor!(G, [(:r, [1, 50, 100])])

sim!([G], [GG]; duration = 100ms)

true
