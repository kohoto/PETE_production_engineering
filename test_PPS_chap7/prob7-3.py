import WellboreFlow as wf
import math as m
from matplotlib import pyplot as plt

if __name__ == '__main__':
    """
        problem 7-3: 1phase gas
    """
    # appendix B: two phase reservoir
    # used 0.0006 as roughness (it's not in the prob statement)
    T_surf = 80  # [F] --
    gamma_g = 0.65
    gas = wf.Fluid(oil=0, newtonian=1, B=1.17, dens=0.0, mu=1.72, T=220, ct=1.29e-5, gamma_g=gamma_g, type='misc', sour=0, Sg=0.73)
    # well properties
    well = wf.Well(p_surf=600, d_tube=2.259, l_fluid=15000, angle=90, roughness=0.0006) #TODO: roughness value?
    # other input
    dTdl = 0.02
    qs = range(1, 5000, 10)  # rate

    pwf = []
    for q in qs:
        rate = wf.Rate(q, 'MSCF_d')
        # get
        pwf.append(well.p_surf + wf.get_dp_for_1phase_gas(gas, well, rate, dTdl)[0])

    # plotting
    fig, ax = plt.subplots(tight_layout=True)
    plt.setp(ax, xlabel='q [MSCF/d]', ylabel='pwf [psi]')
    ax.grid()
    ax.plot(qs, pwf)
    ax.annotate("{:.1f}".format(pwf[-1]), (qs[-1], pwf[-1]), textcoords="offset points", xytext=(0, 10), ha='right')
    ax.set_xlim((0, m.ceil(qs[-1] / 1000) * 1000))
    ax.set_ylim((0, m.ceil(pwf[-1] / 1000) * 1000))
    ax.set_title('Problem 7-3: epsilon=0.0006')
    plt.show()