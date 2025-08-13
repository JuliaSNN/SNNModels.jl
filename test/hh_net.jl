E = HH(; N = 3200)
I = HH(; N = 800)
EE = SpikingSynapse(E, E, :ge; μ = 6nS, p = 0.02)
EI = SpikingSynapse(E, I, :ge; μ = 6nS, p = 0.02)
IE = SpikingSynapse(I, E, :gi; μ = 67nS, p = 0.02)
II = SpikingSynapse(I, I, :gi; μ = 67nS, p = 0.02)
P = [E, I]
C = [EE, EI, IE, II]

monitor!(E, [(:v, [1, 10, 100])])
sim!(P, C; dt = 0.01ms, duration = 100ms)

true