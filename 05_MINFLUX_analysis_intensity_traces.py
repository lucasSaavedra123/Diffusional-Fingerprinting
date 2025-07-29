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
def pool_get_trajectory_fingerprint(traj):
    try:
        return get_trajectory_fingerprint(traj)
    except AssertionError:
        pass

def calculate_msd_parameters(traj):
    DELTA_T = 0.000132
    TIME_START = 0.000084
    MAX_T = 0.050

    t_vec, msd, d, betha, precision, goodness_of_fit = traj.temporal_average_mean_squared_displacement(
        log_log_fit_limit=MAX_T,
        limit_type='time',
        bin_width=DELTA_T,
        time_start=TIME_START,
        with_corrections=True
    )

    msd = msd[t_vec < MAX_T]
    t_vec = t_vec[t_vec < MAX_T]

    traj.info['t_vec'] = t_vec
    traj.info['msd'] = msd
    traj.info['d'] = d
    traj.info['betha'] = betha
    traj.info['precision'] = precision
    traj.info['goodness_of_fit'] = goodness_of_fit

def cache_msd_info(traces):
    for trace in traces.copy():
        try:
            calculate_msd_parameters(trace)
        except AssertionError:
            traces.remove(trace)
    return traces

if __name__ == "__main__":
    categories = ['CF®680R-BTX', 'fPEG-Chol', 'CF®680R-BTX(+fPEG-Chol)', 'fPEG-Chol(+CF®680R-BTX)']

    category_to_colors = {
        0:'purple',
        1:'dimgrey',
        2:'darkorange',
        3:'darkgreen',
        4:'black',
    }

    queries = {
        'CF®680R-BTX':{'info.dataset':'BTX680R'},
        'BTX640R':{'info.dataset':'Control'},
        'fPEG-Chol':{'info.dataset':'CholesterolPEGKK114'},
        'CF®680R-BTX(+fPEG-Chol)':{'info.dataset':'Cholesterol and btx', 'info.classified_experimental_condition':'BTX680R'},
        'fPEG-Chol(+CF®680R-BTX)':{'info.dataset':'Cholesterol and btx', 'info.classified_experimental_condition':'fPEG-Chol'},
    }

    for category in categories:
        """Get fingerprints"""
        if not os.path.isfile(f"X_fingerprints_MINFLUX_intensity_{category}.npy"):
            DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')

            fingerprints = []
            labels = []
            fingerprints_ids = []

            queries[category].update({'info.immobile': False})
            query_fingerprints = list(Trajectory._get_collection().find(queries[category], {f'info.fingerprint':1}))

            for state in [0,1,2,3,4]:
                for fingerprint_doc in tqdm(query_fingerprints):
                    if 'fingerprint' in fingerprint_doc['info'] and 'dgn_state_'+str(state) in fingerprint_doc['info']['fingerprint']:
                        fingerprints_info = fingerprint_doc['info']['fingerprint']['dgn_state_'+str(state)]
                        new_fingerprints = [info['fingerprint'] for info in fingerprints_info]
                        fingerprints.extend(new_fingerprints)
                        labels.extend([state] * len(new_fingerprints))
                        fingerprints_ids.extend([(str(fingerprint_doc['_id']), info['sub_trace_i']) for info in fingerprints_info])

            DatabaseHandler.disconnect()

            np.save(f"X_fingerprints_MINFLUX_{category}", np.array(fingerprints))
            np.save(f"y_MINFLUX_{category}", np.array(labels))
            np.save(f"traj_ids_{category}", np.array(fingerprints_ids))

        """Train classifiers to obtain insights"""
        Xdat = np.load(f"X_fingerprints_MINFLUX_{category}.npy")
        Xdat = Xdat[:, :-5]
        ydat = np.load(f"y_MINFLUX_{category}.npy")
        fingerprints_ids = np.load(f"traj_ids_{category}.npy")
        conv_dict = dict(zip(range(5), [f'St. {state}' for state in range(5)]))
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
            category
        ]

        for mi, m in enumerate([m1]):
            m = np.round(m/(np.sum(m,axis=1).reshape(5,1)),2)
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
                yticks=range(0,5),
                xticks=range(0,5),
                # title=f"{title}\nf1:{f1:4.4f}\nacc:{acc:4.4f}",
                xticklabels=[xnames[i] for i in range(0,5)][::-1],
                yticklabels=[ynames[i] for i in range(0,5)][::-1],
                xlabel="Predicted label",
                ylabel="True label",
            )
            ax.set_title(m_titles[mi])
            ax.xaxis.set_ticks_position("bottom")
            fig.autofmt_xdate(rotation=0)
            fig.tight_layout()
            fig.savefig(os.path.join(PROJECT_PATH, 'Graphics/Matplotlib', f"02_MINFLUX_ANALYSIS_{m_titles[mi]}_intensities_levels.svg"))


        print("Plotting LDA projection 1D")
        colors = [matplotlib.colors.to_rgb(category_to_colors[state]) for state in range(5)]
        x = Xdat.copy()
        y = ydat
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
        plt.xticks(fontsize=32, fontweight='regular', fontfamily='arial')
        plt.yticks(fontsize=32, fontweight='regular', fontfamily='arial')

        for i, l, c in zip(
            range(5),
            [f'St. {state}' for state in range(5)],
            [category_to_colors[state] for state in range(5)],
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
        fig.savefig(os.path.join(PROJECT_PATH, 'Graphics/Matplotlib', f"02_LINDISC_{category}_intensities.svg"))
        plt.clf()

        print("Computing ranked feature-plot")
        learn = ML(x, y)
        learn.Feature_rank(numfeats=5, names=np.array(get_feature_names()[:-5]))
        from matplotlib.lines import Line2D

        custom_lines = [Line2D([0], [0], color=category_to_colors[state], lw=4) for state in range(5)]
        plt.legend(custom_lines, [f'St. {state}' for state in range(5)], loc="upper center")
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_PATH, 'Graphics/Matplotlib', f"02_FEATURE_RANKING_{category}_intensities.svg"))
        plt.clf()

        continue

        classification_result = defaultdict(list)
        for (fing_id,sub_trace_id), classification in zip(fingerprints_ids, zip(new_ydat,y_pred_simultaneous)):
            classification_result[fing_id].append({'sub_trace_i': sub_trace_id, 'classification': classification})

        DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')
        for fing_id in tqdm(classification_result):
            trajectories = Trajectory.objects(id=fing_id)
            assert len(trajectories) == 1
            trajectory = trajectories[0]
            trajectory.info['analysis'][f'predictions_on_each_{state}_segments'] = classification_result[fing_id]
            trajectory.save()
        DatabaseHandler.disconnect()

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