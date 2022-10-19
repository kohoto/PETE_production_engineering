import math as m
import numpy as np
import matplotlib.pyplot as plt
""" 
    classes
"""

class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

""" 
    input
"""


class Reservoir:
    def __init__(self, p, a, b, k, h, B, mu):
        self.p = p
        self.a = a  # reservoir length in cross-section direction
        self.b = b
        self.k = k
        self.h = h
        self.B = B
        self.mu = mu


class Well:
    def __init__(self, pwf, rw, skin, x1=0.0, x2=0.0, y0=0.0, z0=0.0):
        self.pwf = pwf
        self.rw = rw
        self.x1 = x1
        self.x2 = x2
        self.y0 = y0
        self.z0 = z0
        self.skin = skin
        self.L = x2 - x1
            


def get_ipr_for_horizontal_oil(model, res, well):
    if model == 'Furui':
        return Furui_horizontal_IPR(res, well)
    elif model == 'Babu':
        return Babu_and_Odeh_horizontal_IPR(res, well)


def Furui_horizontal_IPR(res, well):
    yb = 0.5 * res.a
    well.y0 = yb  # Furui model has to be symmetry. Set the coordinates for Babu model here.
    well.z0 = 0.5 * res.h
    Iani = np.sqrt(res.k.x / res.k.z)
    k_geom = np.sqrt(res.k.x * res.k.z)
    sr = get_sr(res, well)
    return k_geom * res.b * (res.p - well.pwf) / (141.2 * res.mu * res.B * (np.log(res.h * Iani / well.rw / (Iani + 1)) + np.pi * yb / res.h / Iani - 1.224 + well.skin + sr))


def Babu_and_Odeh_horizontal_IPR(res, well):
    if well.y0== 0.0 or well.z0 == 0.0:
        print('input invalid in Babu model')

    Iani = np.sqrt(res.k.x / res.k.z)
    f = lambda x: (1/3 + x * (-1 + x))

    # sr calculation
    lnCH = 6.28 * res.a / (Iani * res.h) * f(well.y0 / res.a) - np.log(np.sin(np.pi * well.z0 / res.h)) - 0.5 * np.log(res.a / Iani / res.h) - 1.088  # (5-25)
    sr = get_sr(res, well)
    return np.sqrt(res.k.y * res.k.z) * res.b * (res.p - well.pwf) / (141.2 * res.mu * res.B * (np.log(np.sqrt(res.a * res.h) / well.rw) + lnCH - 0.75 + sr + well.skin))  # (5-23)


def get_sr(res, well):
    f = lambda x: (1/3 + x * (-1 + x))
    xmid = 0.5 * (well.x1 + well.x2)
    p_xyz = (res.b / well.L - 1) * (np.log(res.h / well.rw) + 0.25 * np.log(res.k.y / res.k.z) - np.log(np.sin(np.pi * well.z0 / res.h)) - 1.84)
    if (res.a / np.sqrt(res.k.x)) >= (0.75 * res.b / np.sqrt(res.k.y)) > (0.75 * res.h / np.sqrt(res.k.z)):
        F = lambda x: -x * (0.145 + np.log(x) - 0.137 * x * x) if x <= 1 else (2 - x) * (0.145 + np.log(2 - x) - 0.137 * (2 - x) ** 2)  # (5-31)
        pp_xy = 2 * res.b * res.b / well.L / res.h * np.sqrt(res.k.z / res.k.x) * (F(well.L /2 / res.b) + 0.5 * (F((4 * xmid + well.L) / 2 / res.b) - F((4 * xmid - well.L) / 2 / res.b)))
        return p_xyz + pp_xy

    elif res.b / np.sqrt(res.k.y) >= 1.33 * res.a / np.sqrt(res.k.x) > res.h / np.sqrt(res.k.z):
        p_y = 6.28 * res.b * res.b / res.a / res.h * np.sqrt(res.k.y * res.k.z) / res.k.x * (f(xmid / res.b) + well.L / 24 / res.b * (well.L / res.b - 3))
        p_xy = (res.b / well.L - 1) * (6.28 * res.a / res.h * np.sqrt(res.k.z / res.k.y))
        return p_xyz + p_y + p_xy


if __name__ == '__main__':
    """
        prep figures
    """

    fig, axs = plt.subplots(2, 1, tight_layout=True)
    plt.setp(axs, xlabel='q [bpd]', ylabel='pwf [psi]')
    for i in range(2):
        axs[i].grid()

    """
        inputs
    """
    pwf = np.array(range(0, 4001, 100))
    #              Reservoir(p, a, b, k, h, B, mu)
    res_dataset = [Reservoir(4000.0, 2000.0, 4000, Vector(10.0, 10.0, 1.0), 50.0, 1.1, 5.0),  # Furui 4.4 L == b
                   Reservoir(4000.0, 2000.0, 4000, Vector(10.0, 10.0, 1.0), 250.0, 1.1, 5.0),  # Furui 4.4 L == b
                   Reservoir(4000.0, 2000.0, 4000, Vector(10.0, 10.0, 1.0), 500.0, 1.1, 5.0),  # Furui 4.4 L == b
                   Reservoir(4000.0, 4000.0, 4000, Vector(50.0, 50.0, 8.0), 200.0, 1.1, 5.0)]  # Babu 4.5
    #               Well(pwf, rw, skin, x1, x2, y0, z0)
    well_dataset = [Well(pwf, 0.25, 10.0, 0.0, 4000),  # y0 = a / 2, z0 = h / 2
                    Well(pwf, 0.25, 10.0, 0.0, 4000),  # y0 = a / 2, z0 = h / 2
                    Well(pwf, 0.25, 10.0, 0.0, 4000),  # y0 = a / 2, z0 = h / 2
                    Well(pwf, 0.5, 0.0, 250, 3750, 2000, 100)]  # Babu 4.5
    models = ['Furui', 'Furui', 'Furui', 'Babu']
    labels = ['Furui, h=50ft', 'Furui, h=250ft', 'Furui, h=500ft', 'Babu']

    titles = ['Furui (data: Problem 4.4)', 'Babu and Odeh model (data: Problem 4.5)']
    fig_id = [0, 0, 0, 1]

    """
        main program
    """
    for model, res, well, i in zip(models, res_dataset, well_dataset, range(len(models))):
        q = get_ipr_for_horizontal_oil(model, res, well)
        # plotting
        axs[fig_id[i]].plot(q, pwf, label=labels[i])
        axs[fig_id[i]].annotate("{:.1f}".format(q[0]), (q[0],pwf[0]), textcoords="offset points", xytext=(0,10), ha='right')
        axs[fig_id[i]].set_xlim((0, m.ceil(q[0] / 1000) * 1000))
        axs[fig_id[i]].set_ylim((0, m.ceil(pwf[-1] / 1000) * 1000))
        axs[fig_id[i]].legend()
        axs[fig_id[i]].set_title(titles[fig_id[i]])

    """
        save figures
    """
    fig.set_size_inches(8.5, 11)  # letter width
    plt.savefig('HW2.png')