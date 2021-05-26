import numpy as np
import random as rd
import scipy
import powerlaw
import random
import pandas_datareader as pdr
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from collections import Counter
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

if not(os.path.exists("model")):
    os.mkdir("model")


def simulation_abm(N, b, eps1, eps2, z0, NIT, choix_bruit):
    # Simulation du processus

    # Calcul des autres paramètres
    deltat = 2 / (b * (N**2))
    a1 = b * eps1
    a2 = b * eps2
    lambdat = np.random.normal(0, 1, NIT)

    # Evolution de la densité d'agent dans l'état 1
    temps = [0]
    z = [z0]
    for t in range(NIT):
        zt = z[-1]
        zsuiv = zt + deltat * (a1 - (a1 + a2) * zt) + \
            np.sqrt(2 * b * deltat * zt * (1 - zt)) * lambdat[t]
        z.append(zsuiv)
        temps.append(temps[-1] + deltat)

    # Calcul du bruit eta (variation du "mood" des noise traders)
    eta = []
    if choix_bruit == "spin noise":
        for t in range(NIT + 1):
            a = rd.random()
            if a < 0.5:
                eta.append(-1)
            else:
                eta.append(1)
    elif choix_bruit == "uniform noise":
        for t in range(NIT + 1):
            eta.append(-1 + 2 * rd.random())
    else:
        print("Erreur dans le choix du modèle définissant le bruit eta.")

    # Calcul de r0 en fonction de la loi de eta, eps1, eps2 (on impose E[v]=1)
    r0 = (eps2 - 1) / eps1
    if choix_bruit == "spin noise":
        r0 = r0 * 1
    elif choix_bruit == "uniform noise":
        r0 = r0 * 2
    else:
        print("Erreur dans le choix du modèle définissant le bruit eta.")

    # Calcul du log-rendement
    r = []
    for t in range(NIT + 1):
        r.append(r0 * (z[t] / (1 - z[t])) * eta[t])

    return temps, r, z


def log_vraisemblance(eps1, eps2, v, minv):
    T = len(v)
    C = -T * np.log(scipy.special.beta(eps1, eps2))
    F = eps2 * T * (np.log(eps2 - 1) - np.log(eps1))
    somme1 = 0
    somme2 = 0
    for t in range(T):
        if v[t] <= 0:
            vt = minv
        else:
            vt = v[t]
            somme1 += np.log(vt)
            somme2 += np.log(((eps2 - 1) / eps1) + vt)
    G = (eps1 - 1) * somme1 - (eps1 + eps2) * somme2
    return C + F + G


def estimation_param(actif, debut, fin, type_prix, api_key):
    """
    Estimation des paramètres sur données réelles entre les instants
    début et fin
    """

    # Extrait rendement

    _, r, _ = extraire_data(actif, debut, fin, type_prix, api_key)

    # Premières grandeurs
    v = np.abs(r)
    moy = np.mean(v)
    NIT = len(v)
    v = v * (1 / moy)  # on se ramène à une espérance égale à 1
    minv = np.min(v[v != 0])

    # Estimation de epsilon 1 et epsilon 2

    # Powerlaw estimator
    print("Estimation powerlaw")
    vsort = np.sort(v)
    # si plus de 5000 valeurs, on en prend 5000. Sinon, on les prend toutes
    vsort = vsort[0:-1:max(int(NIT / 5000), 1)]

    alpha = powerlaw.Fit(vsort).alpha
    eps2_estim_powerlaw = alpha - 1

    # Maximum Log-likelihood Estimator

    # Epsilon 1 au dizième
    print("Estimation epsilon 1 (à l'unité)")
    eps1_list_1 = [1.1 + 1 * k for k in range(10)]  # 1 à 10
    MLE_array_1 = np.array(list(map(lambda eps:
                                    log_vraisemblance(eps, eps2_estim_powerlaw,
                                                      v, minv), eps1_list_1)))
    eps1_estim_1 = eps1_list_1[np.argmax(MLE_array_1)]
    # plt.plot(eps1_list_1, MLE_array_1)
    # plt.show()
    print("Estimation epsilon 1 (au dizième)")
    eps1_list_2 = [eps1_estim_1 - 1 + 0.2 * k for k in range(11)]
    MLE_array_2 = np.array(list(map(lambda eps:
                                    log_vraisemblance(eps, eps2_estim_powerlaw,
                                                      v, minv), eps1_list_2)))
    eps1_estim = eps1_list_2[np.argmax(MLE_array_2)]

    # Epsilon 2 au dizième
    print("Estimation epsilon 2 (à l'unité)")
    eps2_list_1 = [1.1 + 1 * k for k in range(10)]  # 1 à 10
    MLE_array_3 = np.array(list(map(lambda eps:
                                    log_vraisemblance(eps1_estim, eps, v,
                                                      minv), eps2_list_1)))
    eps2_estim_1 = eps2_list_1[np.argmax(MLE_array_3)]
    # plt.plot(eps2_list_1, MLE_array_1)
    # plt.show()
    print("Estimation epsilon 2 (au dizième)")
    eps2_list_2 = [eps2_estim_1 - 1 + 0.2 * k for k in range(11)]
    MLE_array_4 = np.array(list(map(lambda eps:
                                    log_vraisemblance(eps1_estim, eps, v,
                                                      minv), eps2_list_2)))
    eps2_estim = eps2_list_2[np.argmax(MLE_array_4)]

    # Estimation de b par régression linéaire sur le log des autocorrelations
    # estimation de b par hypothèse sur delta_t
    print("Estimation de b")
    # Calcul des autocorrelations simples
    delta_t = 1
    autocor = smt.stattools.acf(v, nlags=min(1000, NIT))

    # Régression linéaire
    imax = 0
    while imax <= (len(autocor) - 2) and autocor[imax + 1] > 0:
        imax = imax + 1
    # autocorrelations négatifs, on inverse le signe et on réitère le processus
    # pour effectuer une régression linéaire
    if imax == 0:
        autocor_corrige = [-x for x in autocor]
        while ((imax <= (len(autocor_corrige) - 2)) and
               (autocor_corrige[imax+1] > 0)):
            imax = imax + 1
            ln_autocor = np.log(autocor_corrige[1:(imax + 1)])
    else:
        ln_autocor = np.log(autocor[1:(imax + 1)])
    X = sm.add_constant(np.arange(1, (imax + 1)))
    ols = sm.OLS(ln_autocor, X).fit()
    if len(ols.params) >= 2:
        tau = ols.params[1]
    else:  # si pas d'ordonnée à l'origine, on prend juste la pente
        tau = ols.params[0]
    # valeur absolue pour prévenir d'éventuelles erreurs
    b_estim = abs((-tau / (eps1_estim + eps2_estim)) / delta_t)
    N_estim = int(np.sqrt(2/(b_estim*delta_t))) + 1

    return eps1_estim, eps2_estim, b_estim, N_estim


def extraire_data(actif, debut, fin, type_prix, api_key):
    """
    Renvoit les log-rendements / prix de l'actif entre les instants début
    et fin pour le type de prix précisé :
    ("close", "high", "low", "open", "adjClose", "adjHigh", "adjLow",
    "adjOpen")
    """

    df = pdr.get_data_tiingo(actif, api_key=api_key)
    df["date"] = df.index.get_level_values(1)
    df["date"] = pd.to_datetime(df["date"])  # au bon format
    df = df[(df["date"] >= debut) & (df["date"] <= fin)]

    prix = df[type_prix].to_numpy()
    r = np.log(df[type_prix]).diff().dropna()

    temps = df["date"].to_numpy()

    return temps, r.to_numpy(), prix


def delta_pm(actif, debut, fin, type_prix, api_key):
    """
    Calcul de delta_pm entre les instants début et fin pour l'actif
    considéré et pour un type de prix
    """
    _, r, _ = extraire_data(actif, debut, fin, type_prix, api_key)
    n = len(r)

    delta_p = 0
    delta_m = 0
    n_valeurs = 0

    for j in range(n-2):
        n_valeurs = n_valeurs + 1
        a = r[j + 1] - r[j]
        if a >= 0:
            delta_p = delta_p + a
        else:
            delta_m = delta_m + a
    delta_p = delta_p / n_valeurs
    delta_m = delta_m / n_valeurs
    return delta_p, delta_m


def construire_set_donnees(simu, reelle, k, shuffle, api_key):
    """
    Construit le set de données

    simu = []
    ou
    simu = [N, b, eps1, eps2, z0, NIT, choix_bruit, nb_simul]
    reelle = [] ou [actif, date début, date fin, type_prix ("close", "high",
    "low", "open", "adjClose", "adjHigh", "adjLow", "adjOpen")]
    k : nombre de période
    shuffle : True, alors on mélange X et y
    """

    X = []
    y = []

    if simu == [] and reelle == []:
        print("Set de données vide.")
    elif simu != []:
        [N, b, eps1, eps2, z0, NIT, choix_bruit, nb_simul] = simu
        delta_t = 2 / (b * (N**2))  # supposé en jour ici
        for sim in range(nb_simul):
            print("Simulation numéro : " + str(sim + 1))
            _, r, _ = simulation_abm(N, b, eps1, eps2, z0, NIT, choix_bruit)
            rt_journalier = []
            # on récolte des données journalières
            for i in range(0, NIT + 1, max(int(1 / delta_t), 1)):
                rt_journalier.append(r[i])
            n = len(rt_journalier)
            for j in range(n - k):
                x = [rt_journalier[q + j] for q in range(k)]
                if rt_journalier[j + k] > rt_journalier[j + k - 1]:
                    label = 1  # rendement augmente
                else:
                    label = -1
                # filtre les simulations ratées
                if not(np.isnan(np.array(x)).any()):
                    X.append(x)
                    y.append(label)

    if reelle != []:
        [actif, debut, fin, type_prix] = reelle
        _, r, _ = extraire_data(actif, debut, fin, type_prix, api_key)
        n = len(r)
        for j in range(n-k):
            x = [r[q+j] for q in range(k)]
            if r[j+k] > r[j+k-1]:
                label = 1
            else:
                label = -1
            X.append(x)
            y.append(label)

    if shuffle:
        to_shuffle = list(zip(X, y))
        random.shuffle(to_shuffle)
        X, y = zip(*to_shuffle)

    return X, y


def creer_modele(X, y, test_size, model_name):
    """
    Création du modèle et fit

    test_size : entre 0 et 1, pourcentage de nos données totales qui
    serviront au test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size)
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])
    clf.fit(X_train, y_train)
    precision = clf.score(X_test, y_test)

    joblib.dump(clf, "model/" + model_name + ".pkl")
    return precision


def gain(debut, fin, clf, k, delta_p, delta_m, s, nb_init, montant, actif,
         type_prix, api_key):
    """
    Calcul les gains sur une période avec une stratégie s

    s : stratégie
    nb_init : nombre d'actif initial
    montant : montant initial
    """

    # Prédiction

    temps, r, prix = extraire_data(actif, debut, fin, type_prix, api_key)

    prix_pred = [prix[0]]
    n_valeurs = 0
    nb_justes = 0

    # jours entre début et fin en tenant compte ouverture du marché
    n = len(temps)

    liste_prediction = [0 for _ in range(k - 1)]
    pred = [r[q] for q in range(k)]  # initialise aux k premiers rendements
    for i in range(k - 1, n - 2):
        x = [r[q] for q in range(i - k + 1, i + 1)]  # k précédents rendements
        evol = clf.predict([x])[0]
        liste_prediction.append(evol)

        n_valeurs += 1

        if evol == 1:  # prévoit une hausse
            pred.append(r[i] + delta_p)
            if r[i + 1] > r[i]:
                nb_justes += 1
        elif evol == -1:
            pred.append(r[i] + delta_m)
            if r[i + 1] < r[i]:
                nb_justes += 1
        else:
            raise ValueError("Unknown prediction. Different from +/- 1 !")

    for i in range(1, n):
        prix_predit = prix[i - 1] * np.exp(pred[i - 1])
        prix_pred.append(prix_predit)

    prec = (100 * nb_justes) / n_valeurs

    # Stratégie

    argent = montant
    liste_argent = [argent for _ in range(k)]
    stock = nb_init
    nb_actifs = [stock for _ in range(k)]

    if s == "naive":
        # on commence quand on a les données des k premiers jours
        for i in range(k, n-1):
            predit = liste_prediction[i - 1]
            if predit == 1 and argent > prix[i - 1]:
                argent = argent - prix[i - 1]
                stock = stock + 1
            elif predit == -1 and stock >= 1:
                argent = argent + prix[i - 1]
                stock = stock - 1
            liste_argent.append(argent)
            nb_actifs.append(stock)
    elif s == "inv_naive":
        for i in range(k, n - 1):
            predit = liste_prediction[i - 1]
            if predit == 1 and stock >= 1:
                argent = argent + prix[i - 1]
                stock = stock - 1
            elif predit == -1 and argent > prix[i - 1]:
                argent = argent - prix[i - 1]
                stock = stock + 1
            liste_argent.append(argent)
            nb_actifs.append(stock)
    else:
        print("Erreur : La stratégie n'existe pas !")
    gain_final = (nb_actifs[-1] * prix[-1] + liste_argent[-1] -
                  montant - nb_init * prix[0])
    return temps, prix, r, prix_pred, pred, prec, liste_argent, nb_actifs,\
        liste_prediction, gain_final
