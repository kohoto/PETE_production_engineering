import WellboreFlow as wf
import math as m
from matplotlib import pyplot as plt

if __name__ == '__main__':
    """
        problem 7-2: 1phase oil
    """
    # reservoir properties
    # appendix A: undersaturated oil reservoir
    # used initial condition for B and mu
    T_btm = 220  # [F]
    # get_oil_density()
    Bo = 1.17
    gamma_o = 28
    gamma_gd = 0.71
    Rs = 250.0
    dens = ((8830 / (131.5 + gamma_o)) + 0.01361 * gamma_gd * Rs) / Bo
    oil = wf.Fluid(oil=1, newtonian=1, B=Bo, dens=dens, mu=1.72, T=220, ct=1.29e-5)

    p_surf = 800
    d_tube = (2 + 7 / 8)  # [in]
    l_fluid = 10000  # [ft]
    well = wf.Well(p_surf=800, d_tube=2.259, l_fluid=10000, angle=25, roughness=0.0006)

    q = wf.Rate(2e3, 'bbl_d')

    # get
    d2 = well.d_tube  # TODO: change this if the pipe size will change at some point
    pwf = well.p_surf + wf.get_dp_for_1phase_oil(oil, well, q, d2)
    print('1phase oil = ' + str(pwf))

    # (d)
    qs = range(500, 10000, 10)  # rate
    dp_f = []
    for q in qs:
        rate = wf.Rate(q, 'bbl_d')
        # get
        dp_f.append(wf.get_dp_friction_for_1phase_oil(rate, oil, well))

    # plotting
    fig, ax = plt.subplots(tight_layout=True)
    plt.setp(ax, xlabel='q [MSCF/d]', ylabel='dp_f [psi]')
    ax.grid()
    ax.plot(qs, dp_f)
    ax.annotate("{:.1f}".format(dp_f[0]), (qs[0], dp_f[0]), textcoords="offset points", xytext=(0, 10), ha='right')
    ax.annotate("{:.1f}".format(dp_f[-1]), (qs[-1], dp_f[-1]), textcoords="offset points", xytext=(0, 10), ha='right')
    ax.set_xlim((0, m.ceil(qs[-1] / 1000) * 1000))
    ax.set_ylim((0, m.ceil(dp_f[-1] / 1000) * 1000))
    ax.set_title('Problem 7-2 (d): frictional pressure drop')
    plt.show()
