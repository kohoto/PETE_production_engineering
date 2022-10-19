import WellboreFlow as wf
if __name__ == '__main__':
    """
        example 7-2 & 4: 1phase oil
    """
    # reservoir properties
    # appendix A: undersaturated oil reservoir

    T_btm = 200  # [F]
    oil = wf.Fluid(oil=1, newtonian=1, B=1.0, dens=1.05 * 62.4, mu=1.2, T=220, ct=1.29e-5)

    p_surf = 800
    well = wf.Well(p_surf=800, d_tube=2.259, l_fluid=1000.0, angle=-40.0,
                roughness=0.001)  # d_tube in [in], l_fluid in [ft]

    q = wf.Rate(1e3, 'bbl_d')

    # get
    d2 = well.d_tube  # TODO: change this if the pipe size will change at some point
    dp = wf.get_dp_for_1phase_oil(oil, well, q, d2)