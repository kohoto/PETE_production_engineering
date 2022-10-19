import copy
import math as m
import numpy as np
from matplotlib import pyplot as plt

""" 
    input
"""

# global: gas id number
C1 = 0;
C2 = 1;
C3 = 2;
iC4 = 3;
nC4 = 4;
iC5 = 5;
nC5 = 6;
nC6 = 7;
nC7 = 8;
nC8 = 9;
N2 = 10;
CO2 = 11;
H2S = 12;
#  ['C1', 'C2', 'C3', 'iC4', 'nC4', 'iC5', 'nC5', 'nC6', 'nC7', 'nC8', 'N2', 'CO2', 'H2S']
comp_mw = np.array([16.04, 30.07, 44.09, 58.12, 58.12, 72.15, 72.15, 86.17, 100.2, 114.2, 28.02, 44.01, 34.08])

gc = 32.17

# unit conversion
ft32bbl = 5.615
d2s = 1 / 86400
s2d = 86400.0
""" 
    classes
"""


class Well:
    def __init__(self, p_surf, d_tube, l_fluid, angle, roughness):
        self.p_surf = p_surf
        self.d_tube = d_tube
        self.l_fluid = l_fluid
        self.angle = angle
        self.roughness = roughness


class Fluid:
    def __init__(self, oil, newtonian, B, dens, mu, ct, p=0.0, T=0.0, comp_yi=0.0, gamma_g=0.0, type='misc', sour=0, Sg=0.0):
        self.oil = oil  # tf value
        self.newtonian = newtonian
        self.B = B
        self.ct = ct

        if oil:
            self.dens = dens
            self.mu = mu
        else:
            self.p = p
            self.T = T
            self.comp_yi = np.array(comp_yi)
            self.mw = sum(self.comp_yi * comp_mw)
            self.type = type
            self.Sg = Sg

            if gamma_g == 0.0:  # not givenspecified by comp_yi
                self.gamma_g = self.get_gas_gravity()
                self.sour = 1 if self.comp_yi[N2] > 0.0 or self.comp_yi[CO2] > 0.0 else 0
            else:
                self.gamma_g = np.array(gamma_g)
                self.sour = sour

            self.Z = self.get_zfactor(p)
            self.dens = self.get_gas_density(p)
            self.mu = self.get_gas_mu(p)

    def get_gas_mu(self, p):
        # TODO: Ma: appparent molecular weight = sum of mw of comp * comp%.
        temp_r = self.T + 460.0
        a = (9.379 + 0.01607 * self.mw) * temp_r ** 1.5 / (209.2 + 19.26 * self.mw + temp_r)  # (4-24)
        b = 3.448 + 986.4 / temp_r + 0.01009 * self.mw  # (4-25)
        c = 2.447 - 0.2224 * b  # (4-26)
        rho_g_gcc = 0.0160185 * self.get_gas_density(p)  # g/cc
        v = a * 1e-4 * np.exp(b * rho_g_gcc ** c)  # (4-23) density = g/cc
        return a * 1e-4 * np.exp(b * rho_g_gcc ** c)  # (4-23) density = g/cc

    def get_gas_density(self, p):
        return p * self.mw / self.Z / 10.73 / (self.T + 460.0)  # lb/ft3

    def get_zfactor(self, p):
        a1 = 0.3265;
        a2 = -1.07;
        a3 = -0.5339;
        a4 = 0.01569;
        a5 = -0.05165;
        a6 = 0.5475;
        a7 = -0.7361;
        a8 = 0.1844;
        a9 = 0.1056;
        a10 = 0.6134;
        a11 = 0.7210;
        [p_pr, Tpr] = self.get_pseudoreduced_properties(p)
        # Tpr = self.get_pseudoreduced_temp()

        z = np.ones(np.size(p))
        for iter in range(5):  # it's more accurate if I iterate more than 5 times, but usually the textbook calculate only onece
            rho_pr = 0.27 * (p_pr / z / Tpr)
            z = 1.0 + rho_pr * (a1 + (a2 + (a3 + (a4 + a5 / Tpr) / Tpr) / Tpr ** 2) / Tpr
                                + rho_pr * ((a6 + (a7 + a8 / Tpr) / Tpr)
                                            - rho_pr ** 3 * a9 * (a7 + a8 / Tpr) / Tpr)) + a10 * (
                        1 + a11 * rho_pr ** 2) * (
                        rho_pr ** 2 / Tpr ** 3) * np.exp(-a11 * rho_pr ** 2)

        return z

    def get_zfactor_slope(self, p):
        return (self.get_zfactor(p + 50) - self.get_zfactor(p - 50)) / (
                self.get_pseudoreduced_prsr(p + 50) - self.get_pseudoreduced_prsr(p - 50))

    def get_pseudoreduced_properties(self, p):
        # compute epsilon if non-hydro carbon exists
        if self.sour:
            epsilon3 = 120.0 * ((self.comp_yi[CO2] + self.comp_yi[H2S]) ** 0.9 - (self.comp_yi[CO2] + self.comp_yi[H2S]) ** 1.6) \
                       + 15 * (self.comp_yi[CO2] ** 0.5 - self.comp_yi[H2S] ** 4.0)

        if np.all(self.comp_yi) == 0.0:  # detailed composition not known but gas gravity known
            if self.sour:
                gamma_g = (self.gamma_g - 0.967 * self.comp_yi[N2] - 1.52 * self.comp_yi[CO2] - 1.18 * self.comp_yi[H2S]) / \
                          (1.0 - self.comp_yi[N2] - self.comp_yi[CO2] - self.comp_yi[H2S])
            else:
                gamma_g = self.gamma_g

            if self.type == 'misc':
                p_pcHC = 677.0 + 15.0 * gamma_g - 35.7 * gamma_g ** 2
                temp_pcHC = 168.0 + 325.0 * gamma_g - 12.5 * gamma_g ** 2  # in R
            elif self.type == 'condensate':
                p_pcHC = 706.0 + 51.7 * gamma_g - 11.1 * gamma_g ** 2
                temp_pcHC = 187.0 + 330.0 * gamma_g - 71.5 * gamma_g ** 2

            if self.sour:
                [p_pcM, temp_pcM] = self.get_pseudo_critical_properties_for_sour_gas(p_pcHC, temp_pcHC)
            else:
                p_pcM = p_pcHC
                temp_pcM = temp_pcHC

        else:  # detailed compostion is known
            comp_p_pcHC = [673, 709, 618, 530, 551, 482, 485, 434, 397, 361, 492, 1072, 1306]
            comp_temp_pcHC = [344, 550, 666, 733, 766, 830, 847, 915, 973, 1024, 227, 548, 673]

            p_pcM = np.sum(np.array(self.comp_yi) * np.array(comp_p_pcHC))
            temp_pcM = np.sum(np.array(self.comp_yi) * np.array(comp_temp_pcHC))

        if self.sour:
            temp_pc = temp_pcM - epsilon3
            p_pc = p_pcM * temp_pc / (temp_pcM + self.comp_yi[H2S] * (1.0 - self.comp_yi[H2S]) * epsilon3)
        else:
            temp_pc = temp_pcM
            p_pc =p_pcM

        return p / p_pc, (self.T + 460.0) / temp_pc

    def get_pseudo_critical_properties_for_sour_gas(self, p_pcHC, temp_pcHC):
        p_pcM = (1 - self.comp_yi[N2] - self.comp_yi[CO2] - self.comp_yi[H2S]) * p_pcHC + 493 * self.comp_yi[
            N2] + 1071 * self.comp_yi[CO2] + 1.306 * self.comp_yi[H2S]
        temp_pcM = (1 - self.comp_yi[N2] - self.comp_yi[CO2] - self.comp_yi[H2S]) * temp_pcHC + 227 * self.comp_yi[
                N2] + 548 * self.comp_yi[CO2] + 672 * self.comp_yi[H2S]
        return p_pcM, temp_pcM


    def get_gas_gravity(self):
        return self.mw / 28.97


class Rate:
    def __init__(self, q, unit):
        if unit is 'bbl_d':
            self.bbl_d = q
            self.MSCF_d = q * 0.00561458
            self.ft3_d = q * 5.61458

        if unit is 'MSCF_d':
            self.MSCF_d = q
            self.ft3_d = q * 0.1781076
            self.bbl_d = q / 0.00561458

        elif unit is 'ft3_d':
            self.ft3_d = q
            self.MSCF_d = q / 0.1781076
            self.bbl_d = q / 5.61458


""" 
    main program
"""


## general
def get_dp_hydrostatic(l_fluid, dens, angle):
    dppe = 0.006939  # = 1/144
    dp = dens * dppe * l_fluid * np.sin(angle / 180 * np.pi)
    return dp


def get_friction_factor(well, q, fluid=0.0, nre=0.0):  # ok
    import math as m
    # check if we have either fluid or nre
    if nre > 0.0 and fluid == 0.0:
        newtonian = 1
    elif nre == 0.0 and fluid:
        velocity = (q.ft3_d / 86400) / (
                m.pi * well.d_tube * well.d_tube / 144 / 4)  # [ft/s], 1day = 86400 sec q = in [ft3_d]
        newtonian = fluid.newtonian
    else:
        print('you can specify either fluid or nre.')

    if nre > 0.0 or newtonian:  # Newtonian Fluid
        if nre == 0.0:  # skip this if wanna evaluate from own nre.
            nre = 39.53695 * m.pi * velocity * well.d_tube * fluid.dens / fluid.mu

        # Chen's eqn for Newtonian fluid
        if nre <= 2100:  # Laminar Flow
            return 16.0 / nre
        else:  # Turbulent Flow
            logterm1 = m.pow(well.roughness, 1.1098) / 2.8257 + m.pow((7.149 / nre), 0.8981)
            logterm2 = well.roughness / 3.7056 - 5.0452 * m.log10(logterm1) / nre
            ff = -4.0 * m.log10(logterm2)
            return 1.0 / (ff * ff)
    else:  # Power law fluid case: PPS book page.641 - 642
        nre = 0.249 * fluid.dens * m.pow(velocity, 2 - fluid.n) * m.pow(well.d_tube, fluid.n) / (
                m.pow(96, fluid.n) * fluid.n * m.pow((3 * fluid.n + 1) / (4 * fluid.n), fluid.n))

        # Solve F(Y) using Newton iteration method to calculate Y
        a = 4 / m.pow(fluid.n, 0.75)
        b = nre
        c = 1 - fluid.n / 2
        d = 0.4 / m.pow(fluid.n, 1.2)

        tolerance = 0.00000000001
        incr = 0.00001
        X1 = 0.00001

        for iter in range(50):
            Y1 = a * c * np.ln(X1) / np.ln(10.0) - X1 ** (-0.5) + a * np.ln(b) / np.ln(10.0) - d
            X2 = X1 + X1 * incr
            Y2 = a * c * np.ln(X2) / np.ln(10.0) - X2 ** (-0.5) + a * np.ln(b) / np.ln(10.0) - d
            m = (Y2 - Y1) / (X2 - X1)
            X2 = (m * X1 - Y1) / m
            if m.abs(X2 - X1) < tolerance:
                return X2
            X1 = X2
        print('Newton iteration method could not converge in power-law friction calcualtion.')


## 1 phase oil
def get_dp_for_1phase_oil(oil, well, q, d2):  # ok
    dpdz_pe = get_dp_hydrostatic(well.l_fluid, oil.dens, well.angle)
    dpdz_f = get_dp_friction_for_1phase_oil(q, oil, well)
    dpdz_ke = get_dp_kinetic_for_1phase_oil(oil.dens, q, well.d_tube, d2)
    print('dp_f = ' + str(dpdz_f))
    print('dp_ke = ' + str(dpdz_ke))
    return dpdz_pe + dpdz_f + dpdz_ke


def get_dp_friction_for_1phase_oil(q, oil, well):  # ok
    if oil.oil:
        velocity = (q.ft3_d / 86400) / (
                m.pi * well.d_tube * well.d_tube / 144 / 4)  # [ft/s], 1day = 86400 sec q = in [ft3_d] - ok
        dpf = oil.dens * well.l_fluid * get_friction_factor(well, q, fluid=oil)
        return 2 * dpf * velocity * velocity / (32.17 * well.d_tube / 12) / 144
    else:
        print('this function is for 1phase oil.')
        return 1


def get_dp_kinetic_for_1phase_oil(dens, q, d1, d2):  # ok
    return 1.53e-8 * dens * q.bbl_d * q.bbl_d * (1 / d2 ** 4 - 1 / d1 ** 4)  # d [in], dens [lbm/ft3]


## 1 phase gas
def get_dp_for_1phase_gas(gas, well, q, dTdl):
    p_up_prev = 0
    for iter in range(1, 10):
        print('dividing well into ' + str(iter) + ' segment...')
        # reset parameters
        p_down = well.p_surf
        T_down = gas.T
        l_seg = well.l_fluid / iter
        for i in range(iter):
            T_up = T_down + dTdl * l_seg
            T_avg = 0.5 * (T_up + T_down)
            p = get_dpdz_for_1phase_gas(gas, well, l_seg, q, p_down, T_avg)
            print(str(p_down) + 'psi => ' + str(p) + ' psi')
            p_down = p
            T_down = T_up

        if abs(p_up_prev - p) < 0.1:
            break
        elif abs(p_up_prev - p) > 0.1 and iter == 10:
            print('get_dp_friction did not converge')
            return 1
        else:
            p_up_prev = p
    return p - well.p_surf


def get_dpdz_for_1phase_gas(gas, well, l_seg, q, p_down, T_avg):
    gas2 = copy.deepcopy(gas)
    gas2.T = T_avg
    Z_avg = gas2.get_zfactor(p_down)
    mu_avg = gas2.get_gas_mu(p_down)  # ps[1] is downstream (upper when producing)

    nre = 20.09 * gas2.gamma_g * q.MSCF_d / (well.d_tube * mu_avg)
    ff = get_friction_factor(well, q, nre=nre)
    s = -0.0375 * gas2.gamma_g * np.sin(well.angle / 180 * np.pi) * l_seg / (Z_avg * (T_avg + 460.0))  # (7-53) T in R
    dp = np.sqrt(np.exp(-s) * p_down * p_down - 2.685e-3 * ff * (Z_avg * (T_avg + 460.0) * q.MSCF_d) ** 2 / np.sin(
        well.angle / 180 * np.pi) / well.d_tube ** 5 * (1 - np.exp(-s)))
    return dp


def get_superficial_and_mixture_properties(oil, q_o, gas, WOR, GOR, Rs, d_tube_inch):
    q_l_bbl_d = q_o.bbl_d * (WOR + oil.B)
    q_g = gas.B * (GOR - Rs) * q_o.bbl_d  # (3-7)
    # compute superficial velocity
    area = 0.25 * np.pi * d_tube_inch * d_tube_inch / 144  # [ft2]
    u_sl = q_l_bbl_d * ft32bbl * d2s / area  # [ft/s]
    u_sg = q_g * d2s / area * gas.Z  # (7-75)  [ft/s]
    u_m = u_sl + u_sg  # (7-94)  [ft/s]
    print(str(u_m))
    lambda_l = u_sl / u_m
    lambda_g = 1 - lambda_l
    mu_m = oil.mu * lambda_l + gas.mu * lambda_g
    dens_m = oil.dens * lambda_l + gas.dens * lambda_g
    mix = Fluid(oil=1, newtonian=1, B=1.17, dens=dens_m, mu=mu_m, ct=1.29e-5)

    return u_sl, u_sg, u_m, mix, lambda_l, lambda_g


def get_dp_for_2phase_oil_gas(well, oil, gas, q_o, WOR, GOR, Rs, model, sigma):
    # get_dp_kinetic
    [u_sl, u_sg, u_m, mix, lambda_l, lambda_g] = get_superficial_and_mixture_properties(oil, q_o, gas, WOR, GOR, Rs, well.d_tube)
    area = np.pi * well.d_tube * well.d_tube / 144 / 4  # [ft2]
    mdot = area * (u_sl * oil.dens + u_sg * gas.dens) * s2d
    # get liquid velocity number (7-95)
    rhosigma = oil.dens / sigma  # sigma = oil-gas surface tension [dynes/cm]
    n_vl = 1.938 * u_sl * rhosigma ** 0.25

    if model == 'm-HB':  # modified Hagedorn Brown method. Upward vertical flow only!
        lb = 1.071 - 0.2218 * u_m * u_m / (well.d_tube / 12)  # (7-91)

        if lambda_l < lb:  # bubble flow
            us = 0.8  # [ft/s] (7-114)
            yl = 1 - 0.5 * (1 + u_m / us - np.sqrt((1 + u_m / us) ** 2 - 4 * u_sg / us))
            dens_avg = (1 - yl) * gas.dens + yl * oil.dens  # lb/ft3 # (7-89)
            mix = Fluid(dens=dens_avg)
            ff = get_friction_factor(well, q, fluid=mix)  # TODO: check the calculation for nre

            dpdz = 1 / 144 * (
                    dens_avg + ff * mdot * mdot / 7.413e10 / well.d_tube ** 5 / dens_avg / yl ** 2)  # TODO: how to calc this m?
            return dpdz
        else:  # other flow
            n_vg = 1.938 * u_sg * rhosigma ** 0.25
            n_d = 120.872 * (well.d_tube / 12) * np.sqrt(rhosigma)
            n_l = 0.15726 * oil.mu / sigma * rhosigma ** -0.25

            cnl = (0.0019 + 0.0322 * n_l - 0.6642 * n_l * n_l + 4.9951 * n_l ** 3) / (
                    1 - 10.0147 * n_l + 33.8696 * n_l * n_l + 277.2817 * n_l ** 3)
            H = n_vl * well.p_surf ** 0.1 * cnl / (n_vg ** 0.575 * 14.7 ** 0.1 * n_d)

            yl_psi = np.sqrt(
                (0.0047 + 1123.32 * H + 729489.64 * H ** 2) / (1 + 1097.1566 * H + 722153.97 * H ** 2))  # (7-105)
            B = n_vg * n_l ** 0.38 / n_d ** 2.14  # (7-106)
            psi = (1.0086 - 69.9473 * B + 2334.3497 * B ** 2 + - 12896.683 * B ** 3) / (
                    1 - 53.4401 * B + 1517.9369 * B ** 2 - 8419.8115 * B ** 3)  # (7-107)

            yl = yl_psi * psi
            if yl < lambda_l:
                yl = lambda_l

            dens_avg = (1 - yl) * gas.dens + yl * oil.dens  # lb/ft3 # (7-89)

            # Reynolds number
            nre = 2.2e-2 * mdot / (well.d_tube / 12) / oil.mu ** yl / gas.mu ** (1 - yl)
            ff = get_friction_factor(well, 0.0, nre=nre)  # if nre is used to get ff, q is not used

            dpdz = 1 / 144 * (
                    dens_avg + ff * mdot * mdot / 7.413e10 / (well.d_tube /12 ) ** 5 / dens_avg)  # TODO: how to get mix_velocity? u_m
            # dpdz2 = 1 / 144 * (dens_avg + ff * mdot * mdot / 7.413e10 / well.d_tube ** 5 / dens_avg + dens_avg * u_m / well.l_fluid)  #TODO: why don't we need kinetic?
            return dpdz

    elif model == 'BB':  # Beggs and Brill method
        downhill = 0
        yl = get_yl_BB(oil, q_o, gas, WOR, GOR, Rs, well.d_tube, well, n_vl, 0)  # 0 is not downhill

        # Payne correction
        if not downhill:
            if well.angle > 0.0:
                yl = 0.924 * yl
            else:
                yl = 0.685 * yl
        print('yl = ' + str(yl))
        dens_avg = (1 - yl) * gas.dens + yl * oil.dens  # lb/ft3 # (7-89)
        # compute dpdz_f
        # no slip friction factor.
        nre = 1488 * mix.dens * u_m * (well.d_tube / 12) / mix.mu
        fn = get_friction_factor(well, q_o, nre=nre)  #TODO: not sure about the rate
        X = np.log(lambda_l / yl / yl)
        S = X / (-0.0523 + 3.182 * X - 0.8725 * X ** 2 + 0.01853 * X ** 4)  # (7-153)
        ftp = fn * np.exp(S)
        dpdz_f = 2 * ftp * mix.dens * u_m ** 2 / (gc * (well.d_tube / 12)) / 144  # 144 is to lbf/ft3 to psi/ft
        print('BB dpdz_f = ' + str(dpdz_f))
        # compute dpdz_pe
        dpdz_pe = get_dp_hydrostatic(well.l_fluid, dens_avg, well.angle)  # TODO: average dens is same as dens_m?
        print('BB dpdz_pe = ' + str(dpdz_pe))
        # compute dpdz_ke
        dpdz_ke = 0.0
        # Ek = u_m * u_sg * mix.dens / gc / well.p_surf
        # dpdz_ke = (dpdz_f + dpdz_pe) / (1 - Ek)
        return dpdz_pe + dpdz_f + dpdz_ke
    else:
        print('model is not specified, or the name is wrong [model is not specified]')
        return 1


def get_yl_BB(oil, q_o, gas, WOR, GOR, Rs, d_tube, well, n_vl, downhill, lambda_l=0.0, nfr=0.0):
    if lambda_l == 0.0 and nfr == 0.0:  # if not calculated
        [u_sl, u_sg, u_m, mix, lambda_l, lambda_g] = get_superficial_and_mixture_properties(oil, q_o, gas, WOR, GOR, Rs, well.d_tube)
        #u_m = 7.641
        nfr = u_m * u_m / 32.28 / (well.d_tube / 12)
        lambda_l = u_sl / u_m  # no slip holdup
        #lambda_l = 0.852

    L1 = 316 * lambda_l ** 0.302
    L2 = 0.0009252 * lambda_l ** -2.4684
    L3 = 0.10 * lambda_l ** -1.4516
    L4 = 0.5 * lambda_l ** -6.738

    if lambda_l >= 0.01 and L2 < nfr <= L3:  # transition flow (7-137)
        flow_regime = 'transition'
        # get holdup for segregated flow and intermediate flow
        yl_seg = get_yl_BB(oil, q_o, gas, WOR, GOR, Rs, well.d_tube, well, n_vl, 0, lambda_l=0.005, nfr=L1 - 1)
        yl_int = get_yl_BB(oil, q_o, gas, WOR, GOR, Rs, well.d_tube, well, n_vl, 0, lambda_l=0.3, nfr=L1)
        # compute coefficients
        A = (L3 - nfr) / (L3 - L2)
        B = 1 - A

        return A * yl_seg + B * yl_int
    else:
        if (lambda_l < 0.01 and nfr < L1) or (lambda_l >= 0.01 and nfr < L2):  # segragated flow (7-136)
            flow_regime = 'segragated'
            a = 0.98;
            b = 0.4846;
            c = 0.0868;
            d = 0.011;
            e = -3.768;
            f = 3.539;
            g = -1.614;
        elif (0.01 <= lambda_l < 0.4 and L3 < nfr <= L1) or (
                lambda_l > 0.4 and L3 < nfr <= L4):  # intermittent flow (7-138)
            flow_regime = 'intermittent'
            a = 0.845;
            b = 0.5351;
            c = 0.0173;
            d = 2.96;
            e = 0.305;
            f = -0.4473;
            g = 0.0978;
        elif (lambda_l < 0.4 and L1 <= nfr) or (lambda_l >= 0.4 and L4 < nfr):  # distributed flow (7-139)
            flow_regime = 'distributed'
            a = 1.065;
            b = 0.5824;
            c = 0.0609;

        if downhill:  # for downhill, modify d-g. Use the same coeff for all flow regimes
            d = 4.7;
            e = -3.692;
            f = 0.1244;
            g = -0.5056

        ylo = a * lambda_l ** b / nfr ** c

        if flow_regime == 'distributed' and not downhill:
            psi = 1
        else:
            C = (1 - lambda_l) * np.log(d * lambda_l ** e * n_vl ** f * nfr ** g)  # TODO: is this n_vl in mHB?
            psi = 1 + C * (np.sin(1.8 * (well.angle / 180 * np.pi)) - 0.333 * np.sin(
                1.8 * (well.angle / 180 * np.pi)) ** 3)  # TODO check the angle is fine
        return ylo * psi  # yl




