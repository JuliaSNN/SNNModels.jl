E = IF(; N = 100)


r(t) = get_time(t)Hz
S = PoissonStimulus(E, :ge, p_post = 0.2f0, N_pre = 50, param = 1kHz)
monitor!(E, [:ge])
sim!([E], [EmptySynapse()], [S]; duration = 1000ms)

true