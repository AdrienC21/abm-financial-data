actif = "NFLX"
debut = "2015-01-01"  # année, mois, jour #pour calibration et affichage court
fin = "2021-12-31"
type_prix = "close"

debut_prev = "2021-01-01"  # k premiers jours on n'investit pas
fin_prev = "2021-04-30"  # dernier jour ne compte pas

z0 = 0.5
NIT = 1000
choix_bruit = "spin noise"  # spin noise or uniform noise
nb_simul = 30

k = 7
shuffle = False
model_name = "voting_classifier"
test_size = 0.2

s = "naive"
nb_init = 0
montant = 1000
