import WellboreFlow as wf
import math as m
from matplotlib import pyplot as plt

if __name__ == '__main__':
    """
        problem 7-8: 2phase oilgas - Beggs and Brill's method
    """
    # (b) at bottom
    GOR = 1000
    WOR = 1.0
    Rs = 554.3
    q_o = wf.Rate(1200.0, 'bbl_d')

    p_btm = 3000
    T_btm = 180  # [F]
    well = wf.Well(p_surf=p_btm, d_tube=2.259, l_fluid=1.0, angle=55, roughness=0.0006)
    oil = wf.Fluid(oil=1, newtonian=1, B=1.323, dens=51.0, mu=0.86, T=T_btm, ct=1.25e-5)
    # comp_yi = [0.875, 0.083, 0.021, 0.006, 0.002, 0.003, 0.008, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0]
    gas = wf.Fluid(oil=0, newtonian=1, B=0.00509, dens=0.0, mu=0.0, p=p_btm, T=T_btm, ct=1.25e-5, gamma_g=0.71)
    dp = wf.get_dp_for_2phase_oil_gas(well, oil, gas, q_o, WOR, GOR, Rs, 'BB', sigma=30.0)  # not sure about the sigma value
    print('2phase Beggs and Brill method dp at bottom = ' + str(dp))

    """
        problem 7-7: 2phase oilgas - Hagedorn and Brown correlation
    """
    # (a) at surface
    p_surf = 100 + 14.7
    T_surf = 100  # [F]
    oil = wf.Fluid(oil=1, newtonian=1, B=1.0, dens=41.5, mu=0.58, T=T_surf, ct=1.25e-5)
    # comp_yi = [0.875, 0.083, 0.021, 0.006, 0.002, 0.003, 0.008, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0]
    gas = wf.Fluid(oil=0, newtonian=1, B=1.0, dens=0.0, mu=0.0, p=p_surf, T=T_surf, ct=1.25e-5, gamma_g=0.71)
    # well properties
    well = wf.Well(p_surf=p_surf, d_tube=2.259, l_fluid=1.0, angle=90, roughness=0.0006)
    dp = wf.get_dp_for_2phase_oil_gas(well, oil, gas, q_o, WOR, GOR, Rs, 'm-HB', sigma=30.0)  # not sure about the sigma value
    print('2phase Hagedorn and Brown correlation dp at surface dp = ' + str(dp))

    # (b) at bottom
    p_btm = 3000
    T_btm = 180  # [F]
    oil = wf.Fluid(oil=1, newtonian=1, B=1.325, dens=38.9, mu=0.58, T=T_btm, ct=1.25e-5)
    gas = wf.Fluid(oil=0, newtonian=1, B=1.0, dens=0.0, mu=0.0, p=p_btm, T=T_btm, ct=1.25e-5, gamma_g=0.71)
    # well properties
    well.p_surf = p_btm
    dp = wf.get_dp_for_2phase_oil_gas(well, oil, gas, q_o, WOR, GOR, Rs, 'BB', sigma=30.0)  # not sure about the sigma value
    print('2phase Hagedorn and Brown correlation dp at bottom = ' + str(dp))