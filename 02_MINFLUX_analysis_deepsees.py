# %%
"""
This script shows how functions in this folder may be utilized to compute diffusional
fingerprints and analyze results. The first part simulates four types of
random motion with the functions in RandomWalkSims.py.
The second part then computes the diffusional fingerprints using functions in
Fingerprint_feat_gen.py. (Fitting the HMM model may take some time, and a pre-fitted
model is therefore included here to reduce the runtime of this example code).
Finally, the last section plots some exemplary properties computed using the MLGeneral.py script,
outlining how insights mentioned in the paper may be obtained in code.

Henrik Dahl Pinholt
"""

import matplotlib.pyplot as plt
import matplotlib
from MLGeneral import ML, histogram
import os
from pomegranate import *
import numpy as np
# import multiprocess as mp
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from traj_fingerprint.Features import get_trajectory_fingerprint, get_feature_names
from imblearn.under_sampling import RandomUnderSampler
from CONSTANTS import PROJECT_PATH


def get_colors(num_states):
    cmap = plt.cm.get_cmap('turbo',num_states)
    steps = 1./num_states
    color_set =np.array( [cmap(i) for i in np.arange(0,1,steps)])
    return color_set

if __name__ == "__main__":
    deepsees_states = list(range(8))

    DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')
    trajs = list(Trajectory._get_collection().find({'info.immobile': False}, {f'info.fingerprint':1}))
    DatabaseHandler.disconnect()

    fingerprints = []
    labels = []

    for traj_info in tqdm(trajs):
        for state in deepsees_states:
            if 'fingerprint' in traj_info['info'] and f'deepsees_segmentation_state_{state}' in traj_info['info']['fingerprint']:
                fingerprints_info = traj_info['info']['fingerprint'][f'deepsees_segmentation_state_{state}']
                fingerprints_info = [info['fingerprint'] for info in fingerprints_info]
                fingerprints.extend(fingerprints_info)
                labels.extend([state] * len(fingerprints_info))

    """Train classifiers to obtain insights"""
    Xdat = np.array(fingerprints)
    Xdat = Xdat[:, :-5]

    def is_nan_or_none(x):
        import math
        try:
            return x is None or math.isnan(x)
        except TypeError:
            return False

    mask_columnas_validas = []
    for j in range(Xdat.shape[1]):  # iteramos columnas
        col = Xdat[:, j]
        contiene_nan = any(is_nan_or_none(x) for x in col)
        mask_columnas_validas.append(not contiene_nan)

    Xdat = Xdat[:, mask_columnas_validas]
    
    ydat =  np.array(labels)
    conv_dict = dict(zip(range(len(deepsees_states)), list(deepsees_states)))
    ydat = np.array([conv_dict[i] for i in ydat])

    print("Computing confusion matrix")
    X_train, X_test, y_train, y_test = train_test_split(
        Xdat, ydat, test_size=0.2, random_state=42
    )
    rus = RandomUnderSampler(replacement=False, random_state=42)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    learn = ML(X_train, y_train)
    learn.Train(algorithm='Boost', plot=False)
    y_pred_isolated = learn.Predict(ML(X_test, y_test, center=False))[0]
    y_pred_isolated = [learn.to_string[i] for i in y_pred_isolated]
    m1 = confusion_matrix(y_test, y_pred_isolated)

    m_titles = [
        "Multi-State",
    ]
    """
    for mi, m in enumerate([m1]):
        m = np.round(m/(np.sum(m,axis=1).reshape(2,1)),2)
        xnames = learn.to_string
        ynames = learn.to_string
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.matshow(m, cmap="Blues")
        for i in range(m.shape[0]):
            for j in range(m.shape[0]):
                if m[i, j] < np.max(m) / 2:
                    ax.text(j, i, m[i, j], ha="center", color="black")
                else:
                    ax.text(j, i, m[i, j], ha="center", color="white", fontsize=12)
        ax.set(
            yticks=range(0,2),
            xticks=range(0,2),
            # title=f"{title}\nf1:{f1:4.4f}\nacc:{acc:4.4f}",
            xticklabels=[xnames[i] for i in range(0,2)][::-1],
            yticklabels=[ynames[i] for i in range(0,2)][::-1],
            xlabel="Predicted label",
            ylabel="True label",
        )
        ax.set_title(m_titles[mi])
        ax.xaxis.set_ticks_position("bottom")
        fig.autofmt_xdate(rotation=0)
        fig.tight_layout()
        fig.savefig(os.path.join(PROJECT_PATH, 'Graphics/Matplotlib', f"02_MINFLUX_ANALYSIS_{mi}_{state}_DEEPSEES.svg"))
    """
    """
    print("Computing LDA projection 3D bubbles")
    learn.Reduce(n_components=1, method="lin")

    MLfig = plt.figure(figsize=(6, 6))
    MLax = MLfig.add_subplot(1, 1, 1, projection="3d")
    learn.ProjectPlot(axis=MLax, colors=[category_to_colors[category] for category in categories])
    MLfig.tight_layout()
    MLfig.savefig("3Dbubbles_fingerprints", dpi=500)
    """
    for selected_i, (x,y) in enumerate([[Xdat, ydat]]):
        print("Plotting LDA projection 1D")
        colors = [matplotlib.colors.to_rgb('red') for _ in deepsees_states]
        cbins = 4  # Discretizes the interpolation into bins
        cmap_name = "my_list"
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=cbins)
        norm = matplotlib.colors.Normalize(vmin=-10.0, vmax=10.0)
        numfeats = 4
        #x = x[:,np.logical_not(np.isnan(x).any(axis=0))]
        learn = ML(x, y)
        learn.Reduce("lin", n_components=1)

        learn.clf = learn.T
        sort = np.argsort(np.abs(learn.clf.coef_[0]))
        normweight = np.abs(learn.clf.coef_[0][sort])[::-1][:numfeats] / np.max(
            np.abs(learn.clf.coef_[0][sort])[::-1][:numfeats]
        )


        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plt.xticks(fontsize=32, fontweight='regular', fontfamily='arial')
        plt.yticks(fontsize=32, fontweight='regular', fontfamily='arial')

        for i, l, c in zip(
            range(len(deepsees_states)),
            list(deepsees_states),
            get_colors(len(deepsees_states)),
        ):
            print(c)
            center, count, sy = histogram(
                learn.X[learn.y == i][:, 0],
                color=c,
                bars=True,
                ax=ax,
                bins=25,
                alpha=0.7,
                range=(-20, 20),
                normalize=True,
                elinewidth=2,
                capsize=2,
                remove0=True,
                legend=l,
            )

            ax.set_xlim([-20,20])
            ax.axvline(learn.X[learn.y == i][:, 0].mean(),color=c,linestyle='--', linewidth=3)
        fig.savefig(os.path.join(PROJECT_PATH, 'Graphics/Matplotlib', f"02_LINDISC_{selected_i}_{state}_deepsees.svg"))
        plt.clf()

        print("Computing ranked feature-plot")
        learn = ML(x, y)
        feature_names = get_feature_names()[:-5]
        feature_names = np.array(feature_names)[mask_columnas_validas]
        features_ranked, ranking_score = learn.Feature_rank(numfeats=5, names=np.array(feature_names), return_ranking=True)

        fig, ax = plt.subplots(5, 1, figsize=(6, 11))

        for rank_i, feature in enumerate(features_ranked):
            minT = np.min(x[:, feature])
            maxT = np.max(x[:, feature])

            nbins = int(np.ceil(np.log2(len(x[:, feature])) + 1))

            for cat_value in np.unique(y):
                dat = x[:, feature][y == cat_value]

                ax[rank_i].hist(
                    dat,
                    bins=100,
                    range=(minT, maxT),
                    density=True,
                    alpha=0.75,
                    orientation='vertical',
                    color=get_colors(len(deepsees_states))[cat_value]
                )

                ax[rank_i].set_ylabel(feature_names[feature], fontsize=30)
                ax[rank_i].set_xticks([])
                ax[rank_i].set_yticks([])

                ax[rank_i].text(
                    0.98, 0.95,              # Cerca de la esquina superior derecha
                    str(np.round(ranking_score[rank_i],2)),         # El texto que querés mostrar
                    ha='right',              # Alineado horizontal: derecha
                    va='top',                # Alineado vertical: arriba
                    fontsize=30,
                    transform=ax[rank_i].transAxes   # ← Usa sistema de coordenadas del eje (0–1)
                )

        ax[-1].set_xlabel("Normalized Feature Value", fontsize=30)
        #custom_lines = [Line2D([0], [0], color=category_to_colors[category], lw=4) for category in cats]
        #plt.legend(custom_lines, cats, loc="upper center")
        #plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_PATH, 'Graphics/Matplotlib', f"03_FEATURE_RANKING_{selected_i}_{state}_deepsees.svg"))
        plt.clf()
