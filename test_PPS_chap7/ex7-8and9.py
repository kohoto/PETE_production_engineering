import WellboreFlow as wf

if __name__ == '__main__':
    # 2phase test
    WOR = 0.0
    T_btm = 180  # [F] ---
    T_surf = 175  # [F] ---
    p_surf = 800
    oil = wf.Fluid(oil=1, newtonian=1, B=1., dens=0.8 * 62.4, mu=2.0, ct=1.29e-5)
    #         [C1,    C2,     C3,     iC4,    nC4,    iC5,    nC5,    nC6,    nC7, nC8, N2,     CO2,   H2S]
    comp_yi = [0.741, 0.0246, 0.0007, 0.0005, 0.0003, 0.0001, 0.0001, 0.0005, 0.0, 0.0, 0.0592, 0.021, 0.152]
    gas = wf.Fluid(oil=0, newtonian=1, B=1.5, dens=0.0, mu=0.0, ct=1.29e-5, p=p_surf, T=T_surf, comp_yi=comp_yi, gamma_g=0.709,
                type='misc', Sg=0.0)
    # well properties
    # If OD = (2 + 7 / 8) then ID = 2.259
    well = wf.Well(p_surf=p_surf, d_tube=2.259, l_fluid=1, angle=90, roughness=0.0006)  # d_tube in [in]
    q_o = wf.Rate(2000.0, 'bbl_d')
    GOR = 1e6 / q_o.bbl_d
    """
        example 7-9: 2phase oilgas - Beggs and Brill's method
    """
    dp = wf.get_dp_for_2phase_oil_gas(well, oil, gas, q_o, WOR, GOR, 'BB', sigma=30.0)
    print('2phase Beggs and Brill method dpdz = ' + str(dp))
    """
        example 7-8: 2phase oilgas - modified Hagedorn and Brown correlation
    """
    dp = wf.get_dp_for_2phase_oil_gas(well, oil, gas, q_o, WOR, GOR, 'm-HB', sigma=30.0)
    print('2phase Hagedorn and Brown correlation dpdz =' + str(dp))

