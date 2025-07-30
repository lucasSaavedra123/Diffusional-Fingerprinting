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
from traj_fingerprint.Features.Fingerprint_feat_gen import ThirdAppender
from MLGeneral import ML, histogram
import os
from pomegranate import *
import numpy as np
# import multiprocess as mp
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from traj_fingerprint.Features import get_trajectory_fingerprint, get_feature_names
from multiprocessing import Pool
from imblearn.under_sampling import RandomUnderSampler
from CONSTANTS import PROJECT_PATH
import umap
from collections import defaultdict


if __name__ == "__main__":
    categories = ['CF®680R-BTX(+fPEG-Chol)', 'fPEG-Chol(+CF®680R-BTX)']
    states = ['normal', 'directed', 'confinement', 'subdifussive']
    for state in states:
        new_categories = [
            'CF®680R-BTX(+fPEG-Chol) NoMisclassified',
            'CF®680R-BTX(+fPEG-Chol) Misclassified',
            'fPEG-Chol(+CF®680R-BTX) NoMisclassified',
            'fPEG-Chol(+CF®680R-BTX) Misclassified',
        ]

        category_to_colors = {
            'CF®680R-BTX(+fPEG-Chol) NoMisclassified':'darkgray',
            'CF®680R-BTX(+fPEG-Chol) Misclassified':'purple',
            'fPEG-Chol(+CF®680R-BTX) NoMisclassified':'darkgreen',
            'fPEG-Chol(+CF®680R-BTX) Misclassified':'darkorange',
        }

        category_to_colors = {
            'NoMisclassified':'darkgray',
            'Misclassified':'purple',
        }

        categories_labels = ['Misclassified', 'NoMisclassified']

        """Get fingerprints"""
        if not os.path.isfile(f"X_fingerprints_MINFLUX_anomaly_{state}.npy"):
            DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')

            queries = {
                'CF®680R-BTX(+fPEG-Chol)':{'info.dataset':'Cholesterol and btx', 'info.classified_experimental_condition':'BTX680R'},
                'fPEG-Chol(+CF®680R-BTX)':{'info.dataset':'Cholesterol and btx', 'info.classified_experimental_condition':'fPEG-Chol'},
            }

            fingerprints = []
            labels = []
            #['analysis']['fingerprint_anomaly'][0/1]
            for category_id, category in enumerate(categories):
                query_fingerprints = Trajectory._get_collection().find(queries[category],
                    {
                        f'info.analysis.predictions_on_each_{state}_segments': 1,
                        f'info.fingerprint.{state}': 1
                    })

                for fingerprint in tqdm(query_fingerprints):
                    if 'analysis' in fingerprint['info'] and f'predictions_on_each_{state}_segments' in fingerprint['info']['analysis']:
                        for sub_category_i, sub_category in enumerate(categories_labels):
                            new_fingerprints = []

                            for fingerprint_classification in fingerprint['info']['analysis'][f'predictions_on_each_{state}_segments']:
                                raw_fingerprint = [f for f in fingerprint['info']['fingerprint'][state] if f['sub_trace_i']==int(fingerprint_classification['sub_trace_i'])][0]['fingerprint']

                                if sub_category == 'Misclassified' and fingerprint_classification['classification'][0] != fingerprint_classification['classification'][1]:
                                    new_fingerprints.append(raw_fingerprint)
                                if sub_category == 'NoMisclassified' and fingerprint_classification['classification'][0] == fingerprint_classification['classification'][1]:
                                    new_fingerprints.append(raw_fingerprint)

                            fingerprints.extend(new_fingerprints)
                            labels.extend([new_categories.index(category+" "+sub_category)] * len(new_fingerprints))

            DatabaseHandler.disconnect()

            np.save(f"X_fingerprints_MINFLUX_anomaly_{state}", np.array(fingerprints))
            np.save(f"y_MINFLUX_anomaly_{state}", np.array(labels))

        """Train classifiers to obtain insights"""
        Xdat = np.load(f"X_fingerprints_MINFLUX_anomaly_{state}.npy")[:,:-5]
        ydat = np.load(f"y_MINFLUX_anomaly_{state}.npy")
        conv_dict = dict(zip(range(len(new_categories)), list(new_categories)))
        ydat = np.array([conv_dict[i] for i in ydat])

        print("Computing confusion matrix")
        selected_1 = (ydat == new_categories[0]) | (ydat == new_categories[1])
        X_train, X_test, y_train, y_test = train_test_split(
            Xdat[selected_1], ydat[selected_1], test_size=0.2, random_state=42
        )
        rus = RandomUnderSampler(replacement=False, random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        learn = ML(X_train, y_train)
        learn.Train(algorithm='Boost', plot=False)
        y_pred_btx = learn.Predict(ML(X_test, y_test, center=False))[0]
        y_pred_btx = [learn.to_string[i] for i in y_pred_btx]
        m1 = confusion_matrix(y_test, y_pred_btx)

        selected_2 = (ydat == new_categories[2]) | (ydat == new_categories[3])
        X_train, X_test, y_train, y_test = train_test_split(
            Xdat[selected_2], ydat[selected_2], test_size=0.2, random_state=42
        )
        rus = RandomUnderSampler(replacement=False, random_state=42)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        learn = ML(X_train, y_train)
        learn.Train(algorithm='Boost', plot=False)
        y_pred_chol = learn.Predict(ML(X_test, y_test, center=False))[0]
        y_pred_chol = [learn.to_string[i] for i in y_pred_chol]
        m2 = confusion_matrix(y_test, y_pred_chol)

        m_titles = [
            "BTX",
            "fPEG-Chol"
        ]

        for mi, m in enumerate([m1,m2]):
            m = np.round(m/(np.sum(m,axis=1).reshape(2,1)),2)
            xnames = learn.to_string
            xnames = [xnames[name_i].split(' ')[-1] for name_i, _ in enumerate(xnames)]
            ynames = learn.to_string
            ynames = [ynames[name_i].split(' ')[-1] for name_i, _ in enumerate(ynames)]

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
            fig.savefig(os.path.join(PROJECT_PATH, 'Graphics/Matplotlib', f"03_MINFLUX_ANALYSIS_{mi}_{state}.svg"))
        """
        print("Computing LDA projection 3D bubbles")
        learn.Reduce(n_components=1, method="lin")

        MLfig = plt.figure(figsize=(6, 6))
        MLax = MLfig.add_subplot(1, 1, 1, projection="3d")
        learn.ProjectPlot(axis=MLax, colors=[category_to_colors[category] for category in categories])
        MLfig.tight_layout()
        MLfig.savefig("3Dbubbles_fingerprints", dpi=500)
        """
        for selected_i, (x,y,cats) in enumerate([[Xdat[selected_1], ydat[selected_1], new_categories[:2]], [Xdat[selected_2], ydat[selected_2], new_categories[-2:]]]):
            print("Plotting LDA projection 1D")
            cats = [cat.split(' ')[-1] for cat in cats]
            colors = [matplotlib.colors.to_rgb(category_to_colors[category]) for category in cats]
            cbins = 4  # Discretizes the interpolation into bins
            cmap_name = "my_list"
            cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=cbins)
            norm = matplotlib.colors.Normalize(vmin=-10.0, vmax=10.0)
            numfeats = 4
            x = x[:,np.logical_not(np.isnan(x).any(axis=0))]
            learn = ML(x, y)
            learn.Reduce("lin", n_components=1)

            learn.clf = learn.T
            sort = np.argsort(np.abs(learn.clf.coef_[0]))
            normweight = np.abs(learn.clf.coef_[0][sort])[::-1][:numfeats] / np.max(
                np.abs(learn.clf.coef_[0][sort])[::-1][:numfeats]
            )
            #


            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

            for i, l, c in zip(
                range(len(cats)),
                list(cats),
                [category_to_colors[category] for category in cats],
            ):
                print(c)
                center, count, sy = histogram(
                    learn.X[learn.y == i][:, 0],
                    color=c,
                    bars=True,
                    ax=ax,
                    bins=10,
                    alpha=0.7,
                    range=(-6, 4),
                    normalize=True,
                    elinewidth=2,
                    capsize=2,
                    remove0=True,
                    legend=l,
                )

                ax.axvline(learn.X[learn.y == i][:, 0].mean(),color=c,linestyle='--', linewidth=3)
            fig.savefig(os.path.join(PROJECT_PATH, 'Graphics/Matplotlib', f"03_LINDISC_{selected_i}_{state}.svg"))
            plt.clf()
            print("Computing ranked feature-plot")
            learn = ML(x, y)
            feature_names = get_feature_names()[:-5]
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
                        color=category_to_colors[cat_value.split(' ')[-1]]
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
            plt.savefig(os.path.join(PROJECT_PATH, 'Graphics/Matplotlib', f"03_FEATURE_RANKING_{selected_i}_{state}.svg"))
            plt.clf()

        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        umap_object_0 = umap.UMAP()
        umap_object_1 = umap.UMAP()

        umap_object_0.fit(Xdat[selected_1][ydat[selected_1]=='CF®680R-BTX'])
        umap_object_1.fit(Xdat[selected_1][ydat[selected_1]=='fPEG-Chol'])

        embeddings_0 = umap_object_0.transform(Xdat[selected_1][ydat[selected_1]=='CF®680R-BTX'])
        embeddings_1 = umap_object_1.transform(Xdat[selected_1][ydat[selected_1]=='fPEG-Chol'])

        ax[0].scatter(embeddings_0[:,0], embeddings_0[:,1], c='red')
        ax[1].scatter(embeddings_1[:,0], embeddings_1[:,1], c='grey')

        embeddings_0 = umap_object_0.transform(Xdat[selected_2][new_ydat=='CF®680R-BTX'])
        embeddings_1 = umap_object_1.transform(Xdat[selected_2][new_ydat=='fPEG-Chol'])

        ax[0].scatter(embeddings_0[:,0], embeddings_0[:,1], c='black')
        ax[1].scatter(embeddings_1[:,0], embeddings_1[:,1], c='black')

        plt.savefig(os.path.join(PROJECT_PATH, 'Graphics/Matplotlib', f"02_UMAP.svg"))
        plt.clf()
        """
