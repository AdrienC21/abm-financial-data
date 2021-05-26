import numpy as np
import matplotlib.pyplot as plt
import random as rd
import scipy
from statsmodels.distributions.empirical_distribution import ECDF
import powerlaw
from scipy import hypot, pi, sinc, fabs
from math import gamma, log
from statsmodels.tsa.stattools import acf
from sklearn.linear_model import LinearRegression


def simulation_abm(N, b, eps1, eps2, z0, NIT, choix_bruit):
    #Simulation du processus

    #Calcul des autres paramètres
    deltat = 2/(b*(N**2))
    a1 = b * eps1
    a2 = b * eps2
    lambdat = np.random.normal(0, 1, NIT)

    #Evolution de la densité d'agent dans l'état 1
    temps = [0]
    z = [z0]
    for t in range(NIT):
        zt = z[-1]
        zsuiv = zt + deltat * (a1 - (a1 + a2) * zt) + np.sqrt(2 * b * deltat * zt * (1 - zt)) * lambdat[t]
        z.append(zsuiv)
        temps.append(temps[-1] + deltat)

    #Calcul du bruit eta (variation du "mood" des noise traders)
    eta = []
    if choix_bruit == "spin noise":
        for t in range(NIT+1):
            a = rd.random()
            if a < 0.5:
                eta.append(-1)
            else:
                eta.append(1)
    elif choix_bruit == "uniform noise":
        for t in range(NIT+1):
            eta.append(-1 + 2 * rd.random())
    else:
        print("Erreur dans le choix du modèle définissant le bruit eta.")

    #Calcul de r0 en fonction de la loi de eta, eps1, eps2 (on impose E[v]=1)
    r0 = (eps2 - 1) / eps1
    if choix_bruit == "spin noise":
        r0 = r0 * 1
    elif choix_bruit == "uniform noise":
        r0 = r0 * 2
    else:
        print("Erreur dans le choix du modèle définissant le bruit eta.")

    #Calcul du log-rendement
    r = []
    for t in range(NIT+1):
        r.append(r0 * (z[t] / (1 - z[t])) * eta[t])

    return temps, r, z