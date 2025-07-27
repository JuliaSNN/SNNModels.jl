# Define neurons and synapses in the network
N = 1
E = BallAndStick(
    (150um, 200um);
    N = 1,
    soma_syn = Synapse(DuarteGluSoma, MilesGabaSoma),
    dend_syn = Synapse(EyalGluDend, MilesGabaDend),
    NMDA = EyalNMDA,
    param = AdExSoma(b = 0.0f0, Vr = -50),
)


model = merge_models(Dict(:E => E))
monitor!(model.pop.E, [:v_s, :v_d, :he_s, :h_d, :ge_s, :g_d])

sim!(model = model, duration = 1000ms, dt = 0.125)
model.pop.E.v_s[1] = -50mV
model.pop.E.ge_s[1] = 100nS
integrate!(model.pop.E, model.pop.E.param, 0.125f0)
model.pop.E.ge_s
sim!(model = model, duration = 1000ms, dt = 0.125)


vecplot(model.pop.E, :ge_s, dt = 0.125)
