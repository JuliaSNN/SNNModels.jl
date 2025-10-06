N = 100
E1 = IF(; N = N)
E2 = IF(; N = N)
EE = SpikingSynapse(E1, E2, :ge, LTPParam = vSTDPParameter(), conn=(μ = 0.5, p = 1.0))
for n = 1:E1.N
    connect!(EE, n, n)
end
monitor!([E1, E2], [:fire])
monitor!(EE, [:W])
monitor!(EE, [(:x, [20, 10])], variables = :LTPVars)

for t = 1:N
    E1.v[t] = -40
    E2.v[N-t+1] = -40
    train!([E1, E2], [EE], duration = 0.5ms, dt = 0.125ms)
end

ΔW = getrecord(EE, :W)[end]
x = getrecord(EE, :LTPVars_x)[end]

true