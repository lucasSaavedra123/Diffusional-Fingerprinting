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
from Fingerprint_feat_gen import ThirdAppender
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
import pandas as pd

if __name__ == '__main__':
    DATASET_PATH = './BTX680R_CholesterolPEGKK114.csv'
    DATASETS_LABELS = DATASET_PATH.split('/')[1].split('.')[0].split('_')
    FILE_LABEL = f"{DATASETS_LABELS[0]}_{DATASETS_LABELS[1]}"
    dt = 0.001
    """Generate a data set to compute fingerprints for """
    if not os.path.isfile(f"{FILE_LABEL}_X.pkl") or not os.path.isfile(f"{FILE_LABEL}_y.pkl"):
        table = pd.read_csv(DATASET_PATH)
        
        trajectories_by_label = {}
        for label in DATASETS_LABELS:
            trajectories_by_label[label] = []

        for id in tqdm(table['id'].unique()):
            trajectory_df = table[table['id'] == id].sort_values('t')
            new_array = np.zeros((len(trajectory_df), 3))
            new_array[:,0] = trajectory_df['x']
            new_array[:,1] = trajectory_df['y']
            new_array[:,2] = trajectory_df['t']
            trajectories_by_label[trajectory_df['label'].values[0]].append(new_array)

        outdat, labels = [], []
        for label_index, label in enumerate(DATASETS_LABELS):
            outdat += trajectories_by_label[label]
            labels += len(trajectories_by_label[label]) * [label_index]

        with open(f"{FILE_LABEL}_X.pkl", "wb") as f:
            pickle.dump(outdat, f)
        with open(f"{FILE_LABEL}_y.pkl", "wb") as f:
            pickle.dump(labels, f)


    """Compute fingerprints"""
    if not os.path.isfile(f"{FILE_LABEL}_X_fingerprints.npy"):
        import pickle

        print("Generating fingerprints")
        with open(f"{FILE_LABEL}_X.pkl", "rb") as f:
            traces = pickle.load(f)
        if not os.path.isfile(f"{FILE_LABEL}_HMMjson"):
            steplength = []
            for t in traces:
                x, y = t[:, 0], t[:, 1]
                steplength.append(np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2))
            print("fitting HMM")
            model = HiddenMarkovModel.from_samples(
                NormalDistribution, n_components=4, X=steplength, n_jobs=8, verbose=True, stop_threshold=0.001
            )
            #
            print(model)
            model.bake()
            print("Saving HMM model")

            s = model.to_json()
            f = open(f"{FILE_LABEL}_HMMjson", "w")
            f.write(s)
            f.close()
        else:
            print("loading HMM model")
            s = f"{FILE_LABEL}_HMMjson"
            file = open(s, "r")
            json_s = ""
            for line in file:
                json_s += line
            model = HiddenMarkovModel.from_json(json_s)
            print(model)
        d = []
        for t in traces:
            x, y = t[:, 0], t[:, 1]
            SL = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
            d.append((x, y, t[:, 2], SL, dt))

        #p = mp.Pool(mp.cpu_count())
        print("Computing fingerprints")
        print(f"Running {len(traces)} traces")
        #func = partial(ThirdAppender, model=model)  #
        train_result = []
        for t in tqdm(d):
            train_result.append(ThirdAppender(t, model=model)) 
        np.save(f"{FILE_LABEL}_X_fingerprints", train_result)

    """Train classifiers to obtain insights"""
    Xdat = np.load(f"{FILE_LABEL}_X_fingerprints.npy")
    with open(f"{FILE_LABEL}_y.pkl", "rb") as f:
        ydat = pickle.load(f)
    conv_dict = dict(zip(range(len(DATASETS_LABELS)), DATASETS_LABELS))
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
    fig, ax = plt.subplots(1, 1, figsize=(len(DATASETS_LABELS), len(DATASETS_LABELS)))
    ax.matshow(m, cmap="Blues")
    for i in range(m.shape[0]):
        for j in range(m.shape[0]):
            if m[i, j] < np.max(m) / 2:
                ax.text(j, i, m[i, j], ha="center", color="black")
            else:
                ax.text(j, i, m[i, j], ha="center", color="white", fontsize=12)
    ax.set(
        yticks=range(len(DATASETS_LABELS)),
        xticks=range(len(DATASETS_LABELS)),
        # title=f"{title}\nf1:{f1:4.4f}\nacc:{acc:4.4f}",
        xticklabels=[xnames[i] for i in range(len(DATASETS_LABELS))][::-1],
        yticklabels=[ynames[i] for i in range(len(DATASETS_LABELS))][::-1],
        xlabel="Predicted label",
        ylabel="True label",
    )
    ax.xaxis.set_ticks_position("bottom")
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig("Confusion_matrix")
    print("Computing LDA projection 3D bubbles")
    learn.Reduce(n_components=len(DATASETS_LABELS)-1, method="lin")

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
        range(len(DATASETS_LABELS)),
        DATASETS_LABELS,
        ["darkred", "dimgrey", "darkorange", "darkgreen"][:len(DATASETS_LABELS)],
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
        Xdat[(ydat == DATASETS_LABELS[0]) | (ydat == DATASETS_LABELS[1])],
        ydat[(ydat == DATASETS_LABELS[0]) | (ydat == DATASETS_LABELS[1])],
    )

    learn = ML(Xdat_new, ydat_new)

    learn.Feature_rank(numfeats=3)
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color="darkred", lw=4),
        Line2D([0], [0], color="dimgrey", lw=4),
    ]
    plt.legend(custom_lines, DATASETS_LABELS, loc="upper center")
    plt.tight_layout()
    plt.savefig("Feature_ranking")

