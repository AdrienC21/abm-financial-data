from utils import *
from plot import *
from parameters import *
from config import *

afficher_court(actif, debut, fin)
candlestick(actif, debut, fin)

# Estimation of parameters
eps1_estim, eps2_estim, b_estim, N_estim = estimation_param(actif, debut, fin,
                                                            type_prix, api_key)
delta_p, delta_m = delta_pm(actif, debut, fin, type_prix, api_key)

N = N_estim
b = b_estim
eps1 = eps1_estim
eps2 = eps2_estim
simu = [N, b, eps1, eps2, z0, NIT, choix_bruit, nb_simul]
reelle = [actif, debut, fin, type_prix]

X, y = construire_set_donnees(simu, reelle, k, shuffle, api_key)
precision = creer_modele(X, y, test_size, model_name)
clf = joblib.load("model/" + model_name + ".pkl")  # load the saved the model
print("Précision sur simulation / données réelles pour calibration : "
      "{prec}".format(prec=precision))

temps, prix, r, prix_pred, pred, prec, liste_argent, nb_actifs,\
    liste_prediction, gain_final = gain(debut_prev, fin_prev, clf, k, delta_p,
                                        delta_m, s, nb_init, montant, actif,
                                        type_prix, api_key)

plot_predict_rendement(actif, r, pred, prec, temps)
plot_predict_prix(actif, prix, prix_pred, prec, temps)
print_ecart(r, pred, prix, prix_pred)
plot_gain(temps, liste_argent, nb_actifs, prec, gain_final)
plot_court_et_gain(temps, liste_argent, prix, prix_pred)
