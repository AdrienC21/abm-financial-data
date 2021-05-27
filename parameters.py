"""
Data parameters
"""
actif = "NFLX"
debut = "2015-01-01"  # année, mois, jour pour calibration et affichage court
fin = "2021-12-31"  # fin calibration
type_prix = "close"

"""
ABM parameters
"""
z0 = 0.5  # initial proportion of noise traders
NIT = 1000  # number of agents
choix_bruit = "spin noise"  # "spin noise" or "uniform noise"
nb_simul = 30  # number of simulation to create artificial dataset

"""
AI parameters
"""
debut_prev = "2021-01-01"  # start of simulation
fin_prev = "2021-04-30"  # end of simulation
k = 7  # how many days in the past are used as features ?
shuffle = False  # shuffle train & test dataset ?
model_name = "voting_classifier"  # name of the model
test_size = 0.2

"""
Naive investing strategy
"""
s = "naive"  # "naive" or "inv_naive"
nb_init = 0  # initial amount of stock
montant = 1000  # initial amount of money to trade
