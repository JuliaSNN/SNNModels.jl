G = Rate(; N = 100)
GG = RateSynapse(G, G; μ = 1.2, p = 1.0)
monitor!(G, [:r])

train!([G], [GG]; duration = 100ms)
