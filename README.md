# Agent Based Model (Alfarano) & log-return AI

Implementation of an Agent Based Model on financial data and AI algorithm applied to log-return predictions.

## Quick Overview

This project is based on an article written by Simone Alfaranoa, Friedrich Wagnerb and Thomas Luxa : [Estimation of Agent-Based Models: the case of an Asymmetric Herding Model](https://link.springer.com/article/10.1007/s10614-005-6415-1).

- The idea is to implement this model and to fit the ABM parameters on real financial data using different methods.

- We also want to estimate the change of behaviour of our agents by using this article : [Specialization and herding behavior of trading firms in a financial market](https://www.researchgate.net/publication/237526350_Specialization_and_herding_behavior_of_trading_firms_in_a_financial_market) by Lillo & al.

- Then, the aim of this project is to run many simulations in order to create an artificial (but realistic) dataset in order to train an AI which can predict the evolution of the log-return on a daily basis.

- Once we have great predictions, we can try to implement a naive buy/sell trading strategy.

## Presentation

A presentation has been made (**in french**). It contains all the idea and some images. [Link of the presentation](https://github.com/AdrienC21/abm-financial-data/blob/main/ressources/presentation.pdf)

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
## Usage

