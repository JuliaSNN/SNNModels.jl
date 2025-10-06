E = IF(; N = 3200, param = IFParameter(; El = -49mV))
I = IF(; N = 800, param = IFParameter(; El = -49mV))
EE = SpikingSynapse(E, E, :ge; conn =( μ = 60 * 0.27 / 10, p = 0.02))
EI = SpikingSynapse(E, I, :ge; conn =( μ = 60 * 0.27 / 10, p = 0.02))
IE = SpikingSynapse(I, E, :gi; conn =( μ = -20 * 4.5 / 10, p = 0.02))
II = SpikingSynapse(I, I, :gi; conn =( μ = -20 * 4.5 / 10, p = 0.02))
P = [E, I]
C = [EE, EI, IE, II]

monitor!([E, I], [:fire])
sim!(P, C; duration = 1second)
train!(P, C; duration = 1second)
true