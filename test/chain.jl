N = 3
E = IF(; N = N)
EE = SpikingSynapse(E, E, :ge; μ = 0.5, p = 0.8)
for n = 1:(N-1)
    connect!(EE, n, n + 1, 50)
end
E.I[1] = 30

monitor!(E, (:v, [1, N]))
train!([E], [EE]; duration = 100ms)

true