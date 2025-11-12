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
from pomegranate import *
import numpy as np
import pandas as pd
# import multiprocess as mp
from tqdm import tqdm
from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from traj_fingerprint.Features import get_feature_names
import seaborn as sns

def get_colors(num_states):
    cmap = plt.cm.get_cmap('turbo',num_states)
    steps = 1./num_states
    color_set =np.array( [cmap(i) for i in np.arange(0,1,steps)])
    return color_set

colors = get_colors(8)

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

    feature_names = get_feature_names()[:-5]
    feature_names = np.array(feature_names)[mask_columnas_validas]

    df = pd.DataFrame(Xdat, columns=feature_names)
    df = df.astype(np.float)

    corr_matrix = df.corr()
    corr_unstacked = corr_matrix.abs().unstack().sort_values(ascending=False)

    corr_unstacked = corr_unstacked[corr_unstacked < 1]
    corr_unstacked = corr_unstacked[~corr_unstacked.index.duplicated(keep='first')]

    total_pairs = len(corr_unstacked)

    step = 5
    for start in range(0, total_pairs, step):
        end = start + step
        corr_slice = corr_unstacked.iloc[start:end]
        if corr_slice.empty:
            break

        print(corr_slice)

        num_pairs = len(corr_slice)
        plt.figure(figsize=(18, 12))

        for i, ((var1, var2), corr_val) in enumerate(corr_slice.items(), 1):
            plt.subplot(3, 3, i)
            sns.scatterplot(x=df[var1], y=df[var2], alpha=0.6)

            # Ajuste lineal (línea de correlación)
            slope, intercept = np.polyfit(df[var1], df[var2], 1)
            x_vals = np.linspace(df[var1].min(), df[var1].max(), 100)
            plt.plot(x_vals, slope * x_vals + intercept, color='red', linestyle='--', label='Línea de correlación')

            plt.title(f"{var1} vs {var2}\n r = {corr_val:.2f}")
            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.legend()

        plt.suptitle(f"Pares de variables correlacionadas (posiciones {start+1}–{end})", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'correlation_{start}.jpeg', dpi=300)