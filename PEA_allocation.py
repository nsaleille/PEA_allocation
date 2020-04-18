import pandas as pd
import yfinance as yf
import numpy as np

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

def plot_correlation_matrix(matrix, figsize = (15, 15), annot = False):

    sns.set(font_scale=1, rc={'figure.figsize': figsize})

    new_index = matrix.sum().sort_values(ascending = False).index
    matrix = matrix.reindex(index = new_index, columns = new_index)

    mask = np.triu(np.ones_like(matrix, dtype=np.bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(matrix,
                vmin=-1,
                vmax=1,
                cmap=cmap,
                mask = mask,
                square=True,
                linewidths=.5,
                annot = annot,
                cbar_kws={"shrink": .5})

    return None

def portfolio_describe(w, returns_mean, returns_cov_diag, etf_info, by = 'Markov Weight'):

    df = pd.concat([returns_mean, returns_cov_diag, w], axis = 1)
    df.columns = ['Average Return', 'Standard Dev', 'Markov Weight']
    df['Sharpe Ratio'] = df['Average Return'] / df['Standard Dev']
    df = df.loc[df['Markov Weight'] > 0.01]

    columns = ['Catégorie morningstar', "Classe d'actifs", "Zone géographique", "Ref", "Frais de gestion"]
    df = df.join(etf_info[columns], how = 'left')

    df = df.sort_values(ascending = False, by = by)

    cmap = sns.diverging_palette(10, 150, as_cmap=True, center = 'light')

    df_styler = df.style.format({'Average Return' : '{:,.2%}',
                                 'Standard Dev' : '{:,.2f}',
                                 'Markov Weight' : '{0:.2%}',
                                 'Sharpe Ratio' : '{0:.2f}'})\
                        .bar(subset=["Markov Weight"], color='#FFA07A')\
                        .background_gradient(subset = ['Average Return'], cmap='Greens', low = 0)\
                        .background_gradient(subset = ['Standard Dev'], cmap='Blues')

    return df_styler
