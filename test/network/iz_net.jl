Ne = 800;
Ni = 200;
E = IZ(; N = Ne, param = IZParameter(; a = 0.02, b = 0.2, c = -65, d = 8))
I = IZ(; N = Ni, param = IZParameter(; a = 0.1, b = 0.2, c = -65, d = 2))

EE = SpikingSynapse(E, E, :v; conn = (μ = 0.5, p = 0.8))
EI = SpikingSynapse(E, I, :v; conn = (μ = 0.5, p = 0.8))
IE = SpikingSynapse(I, E, :v; conn = (μ = -1.0, p = 0.8))
II = SpikingSynapse(I, I, :v; conn = (μ = -1.0, p = 0.8))
P = [E, I]
C = [EE, EI, IE, II]

monitor!([E, I], [:fire])
for t = 1:1000
    E.I .= 5randn(Ne)
    I.I .= 2randn(Ni)
    sim!(P, C, [EmptyStimulus()], 1.0f0, Time())
end

true
