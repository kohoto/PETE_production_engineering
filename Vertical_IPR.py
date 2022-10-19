import math as m
import numpy as np
import matplotlib.pyplot as plt
import copy
""" 
    input
"""

# global: gas id number
C1 = 1;C2 = 2;C3 = 3;iC4 = 4;nC4 = 5;iC5 = 6;nC5 = 7;nC6 = 8;nC7 = 9;nC8 = 10;N2 = 11;CO2 = 12;H2S = 13;
comp_mw = np.array([16.04, 30.07, 44.09, 58.12, 58.12, 72.15, 72.15, 86.17, 100.2, 114.2, 28.02, 44.01, 34.08])

""" 
    classes
"""

class Well:
    def __init__(self, pwf, rw, skin):
        self.pwf = pwf
        self.rw = rw
        self.skin = skin


class Reservoir:
    def __init__(self, p, re, k, h, B, mu, phi, Z, T, ct):
        self.p = p
        self.re = re
        self.k = k
        self.h = h
        self.B = B
        self.mu = mu
        self.phi = phi
        self.Z = Z
        self.T = T
        self.ct = ct


class Gas:
    def __init__(self, comp_yi, gamma_g, type, Sg):
        self.comp_yi = np.array(comp_yi)
        self.gamma_g = np.array(gamma_g)
        self.mw = sum(self.comp_yi * comp_mw)
        self.type = type
        self.Sg = Sg


""" 
    main program
"""


def get_ipr_for_1phase_oil(regime, res, well, t=0.0):
    if regime == 'ss':
        # IPR for steady-state single phase oil
        coeff = res.k * res.h / 141.2 / res.B / res.mu
        log_term = np.log(res.re / well.rw) + well.skin
    elif regime == 'pss':
        # IPR for pseudo-steady-state single phase oil
        coeff = res.k * res.h / 141.2 / res.B / res.mu
        log_term = np.log(0.472 * res.re / well.rw) + well.skin
    elif regime == 'transient':
        # IPR for transient single phase oil
        coeff = res.k * res.h / 162.6 / res.B / res.mu
        log_term = np.log10(t) + np.log10(
            res.k / res.phi / res.mu / res.ct / well.rw / well.rw) - 3.23 + 0.87 * well.skin

    return coeff * (res.p - well.pwf) / log_term


def get_ipr_for_2phase_oil_gas(regime, res, well, pb, t=0.0):
    well_abs_flow = copy.copy(well)
    if res.p <= pb:  # Vogel correlation (2phase everywhere)
        well_abs_flow.pwf = 0.0  # get absolute flow
        qomax = get_ipr_for_1phase_oil(regime, res, well_abs_flow, t=t) / 1.8
        q = qomax * (1.0 - (well.pwf / res.p) * (0.2 + 0.8 * (well.pwf / res.p)))
    else:  # generalized Vogel correlation (1phase or 2phase)
        well_abs_flow.pwf = pb
        qb = get_ipr_for_1phase_oil(regime, res, well_abs_flow, t=t)
        well_abs_flow.pwf = 0.0  # get absolute flow
        qv = get_ipr_for_1phase_oil(regime, res, well_abs_flow, t=t) / 1.8
        q = (pb <= well.pwf < res.p) * get_ipr_for_1phase_oil(regime, res, well, t=t) + (well.pwf < pb < res.p) * (qb + qv * (1.0 - (well.pwf / pb) * (0.2 + 0.8 * (well.pwf / pb))))

    return q


def get_ipr_for_1phase_gas(regime, res, well, t=0.0):
    # IPR for steady-state single phase gas
    mu_avg = 0.5 * (res.mu + get_gas_mu(well.pwf, res.T, gas))  # array
    if not regime == 'transient':  # ss or pss
        Z_avg = 0.5 * (res.Z + get_zfactor(well.pwf, res.T, gas))  # array
        coeff = res.k * res.h / 1424 / mu_avg / Z_avg / (res.T + 460.0)
        if regime == 'ss':
            log_term = np.log(res.re / well.rw) + well.skin  # same as 1phase oil
        else:
            log_term = np.log(0.472 * res.re / well.rw) + well.skin  # same as 1phase oil

        return coeff * (res.p ** 2 - well.pwf ** 2) / log_term

    else:  # transient
        cg = get_gas_compressibility(res.p, res.T, gas)
        ct = gas.Sg * cg
        mu_ct = mu_avg * ct
        coeff = res.k * res.h / 1638 / (res.T + 460.0)
        log_term = np.log10(t) + np.log10(
            res.k / res.phi / mu_ct / well.rw / well.rw) - 3.23 + 0.87 * well.skin
        return coeff * delta_m(well, res) / log_term


def get_gas_compressibility(p, temp, gas):
    p_pc = p / get_pseudoreduced_prsr(p, gas)
    Z = get_zfactor(p, temp, gas)
    return 1 / p - 1 / Z / p_pc * get_zfactor_slope(p, temp, gas)


def delta_m(well, res):
    # Eq.(4-72) and (4-73)
    p_avg = 0.5 * (res.p + well.pwf)
    mu_avg = 0.5 * (res.mu + get_gas_mu(well.pwf, res.T, gas))  # array
    Z_avg = 0.5 * (res.Z + get_zfactor(well.pwf, res.T, gas))  # array
    return (well.pwf <= 3000.0) * (res.p * res.p - well.pwf ** 2) / mu_avg / Z_avg + (3000.0 < well.pwf) * (
                2 * p_avg / mu_avg / Z_avg * (res.p - well.pwf))


def get_gas_mu(p, temp, gas):
    #TODO: Ma: appparent molecular weight = sum of mw of comp * comp%.
    temp_r = temp + 460.0
    a = (9.379 + 0.01607 * gas.mw) * temp_r ** 1.5 / (209.2 + 19.26 * gas.mw + temp_r)  # (4-24)
    b = 3.448 + 986.4 / temp_r + 0.01009 * gas.mw  # (4-25)
    c = 2.447 - 0.2224 * b  # (4-26)
    rho_g_gcc = 0.0160185 * get_gas_density(p, temp, gas)  # g/cc
    v = a * 1e-4 * np.exp(b * rho_g_gcc ** c)  # (4-23) density = g/cc
    return a * 1e-4 * np.exp(b * rho_g_gcc ** c)  # (4-23) density = g/cc


def get_gas_density(p, temp, gas):
    Z = get_zfactor(p, temp, gas)
    return p * gas.mw / Z / 10.73 / (temp + 460.0)  # lb/ft3


def get_zfactor(p, temp, gas):
    a1 = 0.3265;a2 = -1.07;a3 = -0.5339;a4 = 0.01569;a5 = -0.05165;a6 = 0.5475;a7 = -0.7361;a8 = 0.1844;a9 = 0.1056;a10 = 0.6134;a11 = 0.7210;
    p_pr = get_pseudoreduced_prsr(p, gas)
    Tpr = get_pseudoreduced_temp(temp, gas)

    z = np.ones(np.size(p))
    for iter in range(30):
        rho_pr = 0.27 * (p_pr / z / Tpr)
        z = 1.0 + rho_pr * (a1 + (a2 + (a3 + (a4 + a5 / Tpr) / Tpr) / Tpr ** 2) / Tpr
                            + rho_pr * ((a6 + (a7 + a8 / Tpr) / Tpr)
                            - rho_pr ** 3 * a9 * (a7 + a8 / Tpr) / Tpr)) + a10 * (1 + a11 * rho_pr ** 2) * (
                        rho_pr ** 2 / Tpr ** 3) * np.exp(-a11 * rho_pr ** 2)

    return z


def get_zfactor_slope(p, temp, gas):
    return (get_zfactor(p + 50, temp, gas) - get_zfactor(p - 50, temp, gas)) / (get_pseudoreduced_prsr(p + 50, gas) - get_pseudoreduced_prsr(p - 50, gas))



def get_pseudoreduced_prsr(p, gas):
    gamma_g = get_gas_gravity(gas)
    if gas.type == 'misc':
        p_pc = 677.0 + 15.0 * gamma_g - 35.7 * gamma_g ** 2
    elif gas.type == 'condensate':
        p_pc = 706.0 + 51.7 * gamma_g - 11.1 * gamma_g ** 2

    if gas.comp_yi[N2] > 0.0 or gas.comp_yi[CO2] > 0.0:
        p_pc = (1 - gas.comp_yi[N2] - gas.comp_yi[CO2] - gas.comp_yi[H2S]) * p_pc + 493 * gas.comp_yi[N2] + 1071 * \
               gas.comp_yi[CO2] + 1.306 * gas.comp_yi[H2S]

    return p / p_pc


def get_pseudoreduced_temp(temp, gas):
    gamma_g = get_gas_gravity(gas)
    if gas.type == 'misc':
        temp_pc = 168.0 + 325.0 * gamma_g - 12.5 * gamma_g ** 2  # in R
    elif gas.type == 'condensate':
        temp_pc = 187.0 + 330.0 * gamma_g - 71.5 * gamma_g ** 2

    if gas.comp_yi[N2] > 0.0 or gas.comp_yi[CO2] > 0.0:
        temp_pc = (1 - gas.comp_yi[N2] - gas.comp_yi[CO2] - gas.comp_yi[H2S]) * temp_pc + 227 * gas.comp_yi[N2] + 548 * \
                  gas.comp_yi[CO2] + 672 * gas.comp_yi[H2S]
    return (temp + 460.0) / temp_pc


def get_gas_gravity(gas):
    sum_mw = 0.0
    for i in range(np.size(gas.comp_yi)):
        sum_mw += comp_mw[i] * gas.comp_yi[i]
    return sum_mw / 28.97


if __name__ == '__main__':
    """
        prep figures
    """
    fig, axs = plt.subplots(3, 2, tight_layout=True)
    plt.setp(axs, xlabel='q [bpd]', ylabel='pwf [psi]')
    for i in range(3):
        for j in range(2):
            axs[i, j].grid()


    """
        oil 1phase
    """
    # reservoir properties
    # appendix A: undersaturated oil reservoir
    res = Reservoir(5651.0, 745, 8.2, 53.0, 1.17, 1.72, 0.19, 0, 0, 1.29e-5)
    pwf = np.array(range(0, 5651, 100))
    well = Well(pwf, 0.25, 0.0)

    # oil 1phase
    q_oil_ss = get_ipr_for_1phase_oil('ss', res, well)
    q_oil_pss = get_ipr_for_1phase_oil('pss', res, well)

    # plotting
    axs[0, 0].plot(q_oil_ss, pwf, label='ss')
    axs[0, 0].plot(q_oil_pss, pwf, label='pss')
    axs[0, 0].annotate("{:.1f}".format(q_oil_ss[0]), (q_oil_ss[0],pwf[0]), textcoords="offset points", xytext=(0,10), ha='right')
    axs[0, 0].annotate("{:.1f}".format(q_oil_pss[0]), (q_oil_pss[0],pwf[0]), textcoords="offset points", xytext=(0,10), ha='left')
    axs[0, 0].set_xlim((0, m.ceil(q_oil_ss[0] / 1000) * 1000))
    axs[0, 0].set_ylim((0, m.ceil(pwf[-1] / 1000) * 1000))
    axs[0, 0].set_title('1p oil ss and pss (data: Appendix A)')
    axs[0, 0].legend()

    time = [100, 200, 300]
    for i in range(len(time)):
        q_oil_trans = get_ipr_for_1phase_oil('transient', res, well, t=time[i])
        axs[0, 1].plot(q_oil_trans, pwf, label="t={:.0f} hr".format(time[i]))
        axs[0, 1].annotate("{:.1f}".format(q_oil_trans[0]), (q_oil_trans[0], pwf[0]), textcoords="offset points",
                           xytext=(5, 5+8*i), ha='center')
        # for the first timestep, set xlim
        if i == 0:
            axs[0, 1].set_xlim((0, m.ceil(q_oil_trans[0] / 1000) * 1000))

    axs[0, 1].set_ylim((0, m.ceil(pwf[-1] / 1000) * 1000))
    axs[0, 1].set_title('1p oil transient (data: Appendix A)')
    axs[0, 1].legend()

    """
        2phase
    """
    # appendix B: two phase reservoir
    # B and mu is at bubble point
    res = Reservoir(4336.0, 745, 13.0, 115.0, 1.46, 0.45, 0.21, 0, 220.0, 1.25e-5)
    pb = 4336.0
    pwf = np.array(range(0, 4336, 100))
    well = Well(pwf, 0.25, 0.0)

    # 2phase
    q_2p_ss = get_ipr_for_2phase_oil_gas('ss', res, well, pb)
    q_2p_pss = get_ipr_for_2phase_oil_gas('pss', res, well, pb)
    # plotting
    axs[1, 0].plot(q_2p_ss, pwf, label='ss')
    axs[1, 0].plot(q_2p_pss, pwf, label='pss')
    axs[1, 0].annotate("{:.1f}".format(q_2p_ss[0]), (q_2p_ss[0],pwf[0]), textcoords="offset points", xytext=(0,10), ha='right')
    axs[1, 0].annotate("{:.1f}".format(q_2p_pss[0]), (q_2p_pss[0],pwf[0]), textcoords="offset points", xytext=(0,10), ha='left')
    axs[1, 0].set_xlim((0, m.ceil(q_2p_pss[0] / 1000) * 1000))
    axs[1, 0].set_ylim((0, m.ceil(pwf[-1] / 1000) * 1000))
    axs[1, 0].set_title('2p oil-gas ss and pss (data: Appendix B)')
    axs[1, 0].legend()


    for i in range(len(time)):
        q_2p_trans = get_ipr_for_2phase_oil_gas('transient', res, well, pb, t=time[i])
        axs[1, 1].plot(q_2p_trans, pwf, label="t={:.0f} hr".format(time[i]))
        axs[1, 1].annotate("{:.1f}".format(q_2p_trans[0]), (q_2p_trans[0], pwf[0]), textcoords="offset points",
                           xytext=(5, 5 + 8 * i), ha='center')
        # for the first timestep, set xlim
        if i == 0:
            axs[1, 1].set_xlim((0, m.ceil(q_2p_trans[0] / 1000) * 1000))

    axs[1, 1].set_title('2p oil-gas transient (data: Appendix B)')
    axs[1, 1].legend()
    axs[1, 1].set_ylim((0, m.ceil(pwf[-1] / 1000) * 1000))

    """
        gas 1phase
    """
    # appendix C: well in natural gas reservoir
    pwf = np.array(range(1000, 4613, 100))
    well.pwf = pwf
    res = Reservoir(4613.0, 745, 0.17, 78.0, 0.0, 0.0249, 0.14, 0.945, 180.0, 1.25e-5)
    # res.mu = 0.0249  # at initial condition
    # res.Z = 0.96  # at initial condition

    comp_yi = [0.875, 0.083, 0.021, 0.006, 0.002, 0.003, 0.008, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0]
    gamma_g = 0.65
    gas = Gas(comp_yi, gamma_g, 'misc', 0.73)

    # gas 1phase
    q_gas_ss = get_ipr_for_1phase_gas('ss', res, well)
    q_gas_pss = get_ipr_for_1phase_gas('pss', res, well)

    # plotting
    axs[2, 0].plot(q_gas_ss, pwf, label='ss')
    axs[2, 0].plot(q_gas_pss, pwf, label='pss')
    axs[2, 0].annotate("{:.1f}".format(q_gas_ss[0]), (q_gas_ss[0],pwf[0]), textcoords="offset points", xytext=(0,10), ha='right')
    axs[2, 0].annotate("{:.1f}".format(q_gas_pss[0]), (q_gas_pss[0],pwf[0]), textcoords="offset points", xytext=(0,10), ha='left')
    axs[2, 0].set_xlim((0, m.ceil(q_gas_pss[0] / 1000) * 1000))
    axs[2, 0].set_ylim((1000, m.ceil(pwf[-1] / 1000) * 1000))
    axs[2, 0].set_xlabel('q [MSCF/d]')
    axs[2, 0].set_ylabel('pwf [psi]')
    axs[2, 0].set_title('1p gas ss and pss (data: Appendix C)')
    axs[2, 0].legend()

    for i in range(np.size(time)):
        q_trans = get_ipr_for_1phase_gas('transient', res, well, t=time[i])
        axs[2, 1].plot(q_trans, pwf, label="t={:.0f} hr".format(time[i]))
        axs[2, 1].annotate("{:.1f}".format(q_trans[0]), (q_trans[0], pwf[0]), textcoords="offset points",
                           xytext=(5, 5 + 8 * i), ha='center')
        # for the first timestep, set xlim
        if i == 0:
            axs[2, 1].set_xlim((0, m.ceil(q_trans[0] / 1000) * 1000))

    axs[2, 1].set_xlabel('q [MSCF/d]')
    axs[2, 1].set_ylabel('pwf [psi]')
    axs[2, 1].set_title('1p gas transient (data: Appendix C)')
    axs[2, 1].set_ylim((1000, m.ceil(pwf[-1] / 1000) * 1000))
    axs[2, 1].legend()

    """
        save figures
    """
    fig.set_size_inches(8.5, 11)  # letter width
    plt.savefig('HW1.png')
