import WellboreFlow as wf
if __name__ == '__main__':
    """
        example 7-5: 1phase gas
    """
    # appendix B: two phase reservoir
    # gas component is ex.(4-3)
    T_btm = 200  # [F] --
    T_surf = 150  # [F] --
    comp_yi = [0.741, 0.0246, 0.0007, 0.0005, 0.0003, 0.0001, 0.0001, 0.0005, 0.0, 0.0, 0.0592, 0.021, 0.152]
    gas = wf.Fluid(oil=0, newtonian=1, B=1.17, dens=0.0, mu=1.72, T=T_surf, ct=1.29e-5, comp_yi=comp_yi,
                type='misc', Sg=0.73)
    # well properties
    # If OD = (2 + 7 / 8) then ID = 2.259
    well = wf.Well(p_surf=800, d_tube=2.259, l_fluid=10000, angle=90,
                roughness=0.0006)  # TODO: what roughness to be used?
    # other input
    dTdl = (T_btm - T_surf) / well.l_fluid
    rate = wf.Rate(2e3, 'MSCF_d')
    # get
    pwf = well.p_surf + wf.get_dp_for_1phase_gas(gas, well, rate, dTdl)