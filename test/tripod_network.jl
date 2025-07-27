network = let
    NE = 400
    NI = 100
    E = Tripod()
    I1 = IF(; N = NI ÷ 2, param = IFParameter(τm = 7ms, El = -55mV))
    I2 = IF(; N = NI ÷ 2, param = IFParameter(τm = 20ms, El = -55mV))
    E_to_I1 = SpikingSynapse(E, I1, :ge, p = 0.2, μ = 15.0)
    E_to_I2 = SpikingSynapse(E, I2, :ge, p = 0.2, μ = 15.0)
    I2_to_E = CompartmentSynapse(
        I2,
        E,
        :d1,
        :hi,
        p = 0.2,
        μ = 5.0,
        param = iSTDPPotential(v0 = -50mV),
    )
    I1_to_E = CompartmentSynapse(
        I1,
        E,
        :s,
        :hi,
        p = 0.2,
        μ = 5.0,
        param = iSTDPRate(r = 10Hz),
    )
    E_to_E_d1 = CompartmentSynapse(
        E,
        E,
        :d1,
        :he,
        p = 0.2,
        μ = 30,
        param = vSTDPParameter(),
    )
    E_to_E_d2 = CompartmentSynapse(
        E,
        E,
        :d2,
        :he,
        p = 0.2,
        μ = 30,
        param = vSTDPParameter(),
    )
    pop = dict2ntuple(@strdict E I1 I2)
    recurrent_norm_d1 = SynapseNormalization(
        E,
        [E_to_E_d1],
        param = MultiplicativeNorm(τ = 100ms),
    )
    recurrent_norm_d2 = SynapseNormalization(
        E,
        [E_to_E_d2],
        param = MultiplicativeNorm(τ = 100ms),
    )
    norm1 = recurrent_norm_d1
    norm2 = recurrent_norm_d2
    syn = dict2ntuple(
        @strdict E_to_E_d1 E_to_E_d2 I1_to_E I2_to_E E_to_I1 E_to_I2 norm1 norm2
    )
    (pop = pop, syn = syn)
end

# background

#
clear_records!([network.pop...])
train!([network.pop...], [network.syn...], duration = 5000ms)
