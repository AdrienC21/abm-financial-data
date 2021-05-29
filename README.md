# Agent Based Model (Alfarano) & log-return AI

Implementation of an Agent Based Model on financial data and an AI algorithm applied to log-return predictions.

## Quick Overview

This project is based on an article written by Simone Alfarano, Friedrich Wagnerb and Thomas Luxa : [Estimation of Agent-Based Models: the case of an Asymmetric Herding Model](https://link.springer.com/article/10.1007/s10614-005-6415-1).

- The idea is to implement the model and to fit the ABM parameters on real financial data using different methods.

- We also want to estimate the change of behaviour of our agents by using this article : [Specialization and herding behavior of trading firms in a financial market](https://www.researchgate.net/publication/237526350_Specialization_and_herding_behavior_of_trading_firms_in_a_financial_market) by Lillo & al.

- Then, the aim of this project is to run many simulations in order to create an artificial (but realistic) dataset in order to train an AI which can predict the evolution of the log-return on a daily basis.

- Once we have great predictions, we can try to implement a naive buy/sell trading strategy.

## Presentation

A presentation has been made (**in french**). It contains all the ideas and some results. [Link of the presentation](https://github.com/AdrienC21/abm-financial-data/blob/main/ressources/presentation.pdf)

[![alt text](https://github.com/AdrienC21/abm-financial-data/blob/main/ressources/crop_presentation.png?raw=true)](https://github.com/AdrienC21/abm-financial-data/blob/main/ressources/presentation.pdf)

## Installation

Clone this repository :

```bash
git clone https://github.com/AdrienC21/abm-financial-data.git
```

Make sure the packages powerlaw and mplfinance are installed. If not, type in a python console :

```python
pip install powerlaw
pip install mpl_finance
```
## How to use

The file utils.py contains all the core functions whereas the file plot.py contains the functions used to plot the results/simulations/candlesticks.

First, you need to enter in config.py a tiingo api key (necessary in order to import financial data) :

```python
api_key = ""  # tiingo api key
```

Then, type in parameters.py all the parameters :

```python
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
```

Execute the file run_model.py.

This will plot candlesticks, fit ABM parameters, create a dataset, make predictions, plot the results and try a basic trading strategy.

## License
[MIT](https://choosealicense.com/licenses/mit/)
