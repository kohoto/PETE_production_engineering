import WellboreFlow as wf
if __name__ == '__main__':
    """
        example 7-3: 1phase oil
    """
    T_btm = 200  # [F]
    oil = wf.Fluid(oil=1, newtonian=1, B=1.0, dens=58.0, mu=1.2, T=220, ct=1.29e-5)
    well = wf.Well(p_surf=800, d_tube=4.0, l_fluid=1.0, angle=0.0, roughness=0.001)  # d_tube in [in], l_fluid in [ft]

    q = wf.Rate(2e3, 'bbl_d')

    # get
    d2 = 2.0  # TODO: change this if the pipe size will change at some point
    dp =  wf.get_dp_for_1phase_oil(oil, well, q, d2)