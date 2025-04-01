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
        'CF®680R-BTX':'darkred',
        'fPEG-Chol':'dimgrey',
        'CF®680R-BTX(+fPEG-Chol)':'darkorange',
        'fPEG-Chol(+CF®680R-BTX)':'darkgreen',
        'BTX640R': 'purple'
    }

    """Get fingerprints"""
    if not os.path.isfile("X_fingerprints_MINFLUX.npy"):
        DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')

        queries = {
            'CF®680R-BTX':{'info.dataset':'BTX680R'},
            'BTX640R':{'info.dataset':'Control'},
            'fPEG-Chol':{'info.dataset':'CholesterolPEGKK114'},
            'CF®680R-BTX(+fPEG-Chol)':{'info.dataset':'Cholesterol and btx', 'info.classified_experimental_condition':'BTX680R'},
            'fPEG-Chol(+CF®680R-BTX)':{'info.dataset':'Cholesterol and btx', 'info.classified_experimental_condition':'fPEG-Chol'},
        }

        fingerprints = []
        labels = []
        fingerprints_ids = []

        for category_id, category in enumerate(categories):
            query_fingerprints = Trajectory._get_collection().find(queries[category], {f'info.fingerprint':1})
            for fingerprint in tqdm(query_fingerprints):
                if 'fingerprint' in fingerprint['info']:
                    del fingerprint['info']['fingerprint']['full']
                    new_fingerprints = [f for f in fingerprint['info']['fingerprint'].values()]
                    for i in range(len(new_fingerprints)):
                        new_fingerprints[i] = [np.NaN if f is None else f for f in new_fingerprints[i]]
                    fingerprints.extend(new_fingerprints)
                    labels.extend([category_id] * len(fingerprint['info']['fingerprint']))
                    fingerprints_ids.extend([str(fingerprint['_id'])] * len(fingerprint['info']['fingerprint']))

        DatabaseHandler.disconnect()

        np.save("X_fingerprints_MINFLUX", np.array(fingerprints))
        np.save("y_MINFLUX", np.array(labels))
        np.save("traj_ids", np.array(fingerprints_ids))

    """Train classifiers to obtain insights"""
    Xdat = np.load("X_fingerprints_MINFLUX.npy")[:,:-5]
    ydat = np.load("y_MINFLUX.npy")
    conv_dict = dict(zip(range(len(categories)), list(categories)))
    ydat = np.array([conv_dict[i] for i in ydat])

    print("Computing confusion matrix")
    selected_1 = (ydat == categories[0]) | (ydat == categories[1])
    X_train, X_test, y_train, y_test = train_test_split(
        Xdat[selected_1], ydat[selected_1], test_size=0.2, random_state=42
    )
    rus = RandomUnderSampler(replacement=False, random_state=42)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    learn = ML(X_train, y_train)
    learn.Train(algorithm='Boost')
    y_pred_isolated = learn.Predict(ML(X_test, y_test, center=False))
    m1 = confusion_matrix(y_test, [learn.to_string[i] for i in y_pred_isolated[0]])

    selected_2 = (ydat == categories[2]) | (ydat == categories[3])
    new_ydat = ydat[selected_2]
    new_ydat[new_ydat == categories[2]] = categories[0]
    new_ydat[new_ydat == categories[3]] = categories[1]

    y_pred_simultaneous = learn.Predict(ML(Xdat[selected_2], new_ydat, center=False))
    m2 = confusion_matrix(new_ydat, [learn.to_string[i] for i in y_pred_simultaneous[0]])

    m_titles = [
        "Both probes isolated",
        "Both probes stained simultaneously"
    ]

    for mi, m in enumerate([m1,m2]):
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
        fig.savefig(os.path.join(PROJECT_PATH, 'Graphics/Matplotlib', f"02_MINFLUX_ANALYSIS_{mi}.svg"))
    """
    print("Computing LDA projection 3D bubbles")
    learn.Reduce(n_components=1, method="lin")

    MLfig = plt.figure(figsize=(6, 6))
    MLax = MLfig.add_subplot(1, 1, 1, projection="3d")
    learn.ProjectPlot(axis=MLax, colors=[category_to_colors[category] for category in categories])
    MLfig.tight_layout()
    MLfig.savefig("3Dbubbles_fingerprints", dpi=500)
    """
    for selected_i, (x,y) in enumerate([[Xdat[selected_1], ydat[selected_1]], [Xdat[selected_2], new_ydat]]):
        print("Plotting LDA projection 1D")
        colors = [matplotlib.colors.to_rgb(category_to_colors[category]) for category in categories[:2]]
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
            range(len(categories[:2])),
            list(categories[:2]),
            [category_to_colors[category] for category in categories[:2]],
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
        fig.savefig(os.path.join(PROJECT_PATH, 'Graphics/Matplotlib', f"02_LINDISC_{selected_i}.svg"))
        plt.clf()
        print("Computing ranked feature-plot")
        learn = ML(x, y)
        learn.Feature_rank(numfeats=5, names=np.array(get_feature_names()[:-5]))
        from matplotlib.lines import Line2D

        custom_lines = [Line2D([0], [0], color=category_to_colors[category], lw=4) for category in categories[:2]]
        plt.legend(custom_lines, categories[:2], loc="upper center")
        plt.tight_layout()
        plt.savefig(os.path.join(PROJECT_PATH, 'Graphics/Matplotlib', f"02_FEATURE_RANKING_{selected_i}.svg"))
