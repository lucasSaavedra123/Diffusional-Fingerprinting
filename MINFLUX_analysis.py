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
from RandomWalkSims import (
    Gen_normal_diff,
    Gen_directed_diff,
    Get_params,
    Gen_confined_diff,
    Gen_anomalous_diff,
)
import matplotlib.pyplot as plt
import matplotlib
from traj_fingerprint.Features.Fingerprint_feat_gen import ThirdAppender
from MLGeneral import ML, histogram
import pickle
import os
from pomegranate import *
from functools import partial
import numpy as np
# import multiprocess as mp
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from traj_fingerprint.Features import get_trajectory_fingerprint
from multiprocessing import Pool
import time

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

if __name__ == "__main__":
    """Compute fingerprints"""
    if not os.path.isfile("X_fingerprints.npy"):
        DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')
        btx_traces = list(Trajectory.objects(info__dataset='BTX680R'))[:50]
        btx_with_chol_traces = list(Trajectory.objects(info__dataset='Cholesterol and btx', info__classified_experimental_condition='BTX680R'))[:50]
        DatabaseHandler.disconnect()

        fingerprints = []
        labels = []

        for trace in btx_traces.copy():
            try:
                calculate_msd_parameters(trace)
                labels.append(0)
            except AssertionError:
                btx_traces.remove(trace)

        for trace in btx_with_chol_traces.copy():
            try:
                calculate_msd_parameters(trace)
            except AssertionError:
                btx_with_chol_traces.remove(trace)
                labels.append(1)

        traces = btx_traces + btx_with_chol_traces

        print("Computing fingerprints and labels")
        print(f"Running {len(traces)} traces")

        """
        With Pool, the process is 4.42 times faster.
        """
        with Pool() as pool, tqdm(total=len(traces)) as pbar:
            for result in pool.imap(pool_get_trajectory_fingerprint, traces):
                fingerprints.append(result)
                pbar.update()
                pbar.refresh()

        np.save("X_fingerprints", np.array(fingerprints))
        np.save("y", np.array(labels))

    """Train classifiers to obtain insights"""
    Xdat = np.load("X_fingerprints.npy")
    with open("y.pkl", "rb") as f:
        ydat = pickle.load(f)
    conv_dict = dict(zip(range(4), ["ND", "DM", "CD", "AD"]))
    ydat = np.array([conv_dict[i] for i in ydat])
    learn = ML(Xdat, ydat)
    learn.Train(algorithm="Logistic")
    print("Computing confusion matrix")
    X_train, X_test, y_train, y_test = train_test_split(
        Xdat, ydat, test_size=0.3, random_state=42
    )
    y_pred = learn.Predict(ML(X_test, y_test, center=False))

    m = confusion_matrix(y_test, [learn.to_string[i] for i in y_pred[0]])

    xnames = learn.to_string
    ynames = learn.to_string
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.matshow(m, cmap="Blues")
    for i in range(m.shape[0]):
        for j in range(m.shape[0]):
            if m[i, j] < np.max(m) / 2:
                ax.text(j, i, m[i, j], ha="center", color="black")
            else:
                ax.text(j, i, m[i, j], ha="center", color="white", fontsize=12)
    ax.set(
        yticks=range(4),
        xticks=range(4),
        # title=f"{title}\nf1:{f1:4.4f}\nacc:{acc:4.4f}",
        xticklabels=[xnames[i] for i in range(4)][::-1],
        yticklabels=[ynames[i] for i in range(4)][::-1],
        xlabel="Predicted label",
        ylabel="True label",
    )
    ax.xaxis.set_ticks_position("bottom")
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig("Confusion_matrix")
    print("Computing LDA projection 3D bubbles")
    learn.Reduce(n_components=3, method="lin")

    MLfig = plt.figure(figsize=(6, 6))
    MLax = MLfig.add_subplot(1, 1, 1, projection="3d")
    learn.ProjectPlot(axis=MLax, colors=["darkred", "dimgrey", "darkorange", "darkgreen"])
    MLfig.tight_layout()
    MLfig.savefig("3Dbubbles_fingerprints", dpi=500)

    print("Plotting LDA projection 1D")

    colors = [
        matplotlib.colors.to_rgb("darkred"),
        matplotlib.colors.to_rgb("dimgrey"),
        matplotlib.colors.to_rgb("darkorange"),
        matplotlib.colors.to_rgb("darkgreen"),
    ]  # R -> G -> B
    cbins = 4  # Discretizes the interpolation into bins
    cmap_name = "my_list"
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=cbins)
    norm = matplotlib.colors.Normalize(vmin=-10.0, vmax=10.0)
    numfeats = 4
    learn = ML(Xdat, ydat)
    learn.Reduce("lin", n_components=1)

    learn.clf = learn.T
    sort = np.argsort(np.abs(learn.clf.coef_[0]))
    normweight = np.abs(learn.clf.coef_[0][sort])[::-1][:numfeats] / np.max(
        np.abs(learn.clf.coef_[0][sort])[::-1][:numfeats]
    )
    #


    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    for i, l, c in zip(
        range(4),
        ["ND", "DM", "CD", "AD"],
        ["darkred", "dimgrey", "darkorange", "darkgreen"],
    ):
        print(c)
        center, count, sy = histogram(
            learn.X[learn.y == i][:, 0],
            color=c,
            bars=True,
            ax=ax,
            bins=10,
            alpha=0.7,
            # range=(-5, 5),
            normalize=True,
            elinewidth=2,
            capsize=2,
            remove0=True,
            legend=l,
        )
    fig.savefig("Lindisc.pdf")

    print("Computing ranked feature-plot between normal and directed motion")

    Xdat_new, ydat_new = (
        Xdat[(ydat == "CD") | (ydat == "DM")],
        ydat[(ydat == "CD") | (ydat == "DM")],
    )

    learn = ML(Xdat_new, ydat_new)

    learn.Feature_rank(numfeats=3)
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color="darkred", lw=4),
        Line2D([0], [0], color="dimgrey", lw=4),
    ]
    plt.legend(custom_lines, ["Confined diffusion", "Directed motion"], loc="upper center")
    plt.tight_layout()
    plt.savefig("Feature_ranking")

