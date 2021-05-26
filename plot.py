import pandas_datareader as pdr
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt
import pandas as pd
from config import *


def afficher_court(actif, debut, fin):
    style.use('ggplot')
    df = pdr.get_data_tiingo(actif, api_key=api_key)
    df["date"] = df.index.get_level_values(1)
    df["date"] = pd.to_datetime(df["date"])  # au bon format
    df = df[(df["date"] >= debut) & (df["date"] <= fin)]

    df['100MovingAv'] = df['close'].rolling(window=100).mean()

    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)

    close, = ax1.plot(df["date"], df['close'], label="Close")
    mvgavg, = ax1.plot(df["date"], df['100MovingAv'], label="Moving Avg")
    # ax2.bar (mais sans légende)
    vol, = ax2.plot(df["date"], df['volume'], label="Volume")

    plt.legend([close, mvgavg, vol], ["Close", "Moving Avg", "Volume"],
               loc="best")

    plt.show()


def candlestick(actif, debut, fin):
    style.use('ggplot')
    df = pdr.get_data_tiingo(actif, api_key=api_key)
    df["date"] = df.index.get_level_values(1)
    df["date"] = pd.to_datetime(df["date"])  # au bon format
    df = df[(df["date"] >= debut) & (df["date"] <= fin)]

    df = df.reset_index(level=[0])  # multi index into columns

    df_ohlc = df['close'].resample('10D').ohlc()
    df_volume = df['volume'].resample('10D').sum()
    df_ohlc = df_ohlc.reset_index()
    df_ohlc['date'] = df_ohlc['date'].map(mdates.date2num)

    fig = plt.figure()
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
    ax1.xaxis_date()

    candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')

    ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)

    plt.show()


def plot_predict_rendement(actif, r, pred, prec, temps):
    plt.plot(temps[1:], r, "r", label="Données réelles " + str(actif))
    plt.plot(temps[1:], pred, "g--", label="Prédictions")
    plt.xlabel("jour")
    plt.ylabel("log-rendement")
    plt.title("Prédictions des log-rendements avec du clustering. "
              "Précision : " + str(prec) + "%")
    plt.legend(loc="best")
    plt.show()


def plot_predict_prix(actif, prix, prix_pred, prec, temps):
    plt.plot(temps, prix, "r", label="Données réelles " + str(actif))
    plt.plot(temps, prix_pred, "g--", label="Prédictions")
    plt.xlabel("jour")
    plt.ylabel("prix")
    plt.title("Prédictions du prix avec du clustering. Précision "
              ": " + str(prec) + "%")
    plt.legend(loc="best")
    plt.show()


def print_ecart(r, pred, prix, prix_pred):
    ecart_r, ecart_abs_r, ecart_p, ecart_abs_p = 0, 0, 0, 0
    n = len(pred)
    for i in range(n):
        ecart_r += pred[i] - r[i]
        ecart_abs_r += abs(pred[i] - r[i])
        ecart_p += prix[i] - prix_pred[i]
        ecart_abs_p += abs(prix[i] - prix_pred[i])
    ecart_r, ecart_abs_r, \
        ecart_p, ecart_abs_p = ecart_r / n, ecart_abs_r / n, \
        ecart_p / n, ecart_abs_p / n
    print("Ecart moyen (rendements) : " + str(ecart_r))
    print("Ecart absolu moyen (rendements) : " + str(ecart_abs_r))
    print("Ecart moyen (prix) : " + str(ecart_p))
    print("Ecart absolu moyen (prix) : " + str(ecart_abs_p))


def plot_gain(temps, liste_argent, nb_actifs, prec, gain_final):
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(temps[:-1], liste_argent, "g", label="Argent")
    axs[0].set_title("Porte-feuille (argent)")
    axs[0].set_xlabel("Jour")
    axs[0].set_ylabel("Argent")
    fig.suptitle("Evolution du porte-feuille. Précision : " + str(prec) +
                 "%, Gain final : " + str(gain_final), fontsize=16)

    axs[1].plot(temps[:-1], nb_actifs, label="Nombre d'actif")
    axs[1].set_title("Porte-feuille (nombre d'actif)")
    axs[1].set_xlabel("Jour")
    axs[1].set_ylabel("Actifs")

    plt.legend(loc="best")
    plt.show()


def plot_court_et_gain(temps, liste_argent, prix, prix_pred):
    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(temps, prix, "r", label="Prix réelle")
    axs[0].plot(temps, prix_pred, "g--", label="Prix prédit")
    axs[0].set_title("Prix réelle / prédictions")
    axs[0].set_xlabel("Jour")
    axs[0].set_ylabel("Prix")
    fig.suptitle("Evolution du court et des gains.", fontsize=16)

    axs[1].plot(temps[:-1], liste_argent, "g", label="Argent")
    axs[1].set_title("Porte-feuille (argent)")
    axs[1].set_xlabel("Jour")
    axs[1].set_ylabel("Argent")

    plt.legend(loc="best")
    plt.show()
