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
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

indexes_to_remove = [get_feature_names().index(label) for label in ['$DCR_{Mean}$']]

if __name__ == "__main__":
    categories = ['CF®680R-BTX', 'fPEG-Chol', 'CF®680R-BTX(+fPEG-Chol)', 'fPEG-Chol(+CF®680R-BTX)']
    states = ['normal', 'directed', 'confinement', 'subdifussive']
    for state in states:
        category_to_colors = {
            'CF®680R-BTX':'purple',
            'fPEG-Chol':'dimgrey',
            'CF®680R-BTX(+fPEG-Chol)':'darkorange',
            'fPEG-Chol(+CF®680R-BTX)':'darkgreen',
            'BTX640R': 'purple'
        }

        """Get fingerprints"""
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
        mean_dcrs = []

        for category_id, category in enumerate(categories):
            queries[category].update({'info.immobile': False})
            query_fingerprints = list(Trajectory._get_collection().find(queries[category], {f'info.fingerprint':1}))
            for fingerprint in tqdm(query_fingerprints):
                if 'fingerprint' in fingerprint['info']:
                    del fingerprint['info']['fingerprint']['full']
                    fingerprints_info = fingerprint['info']['fingerprint'][state]
                    new_fingerprints = [info['fingerprint'] for info in fingerprints_info]
                    new_dcrs = []
                    for info in fingerprints_info:
                        if 'mean_dcr' in info:
                            new_dcrs.append(info['mean_dcr'])
                        else:
                            new_dcrs.append(None)
                    mean_dcrs.extend(new_dcrs)
                    fingerprints.extend(new_fingerprints)
                    labels.extend([category_id] * len(new_fingerprints))
                    fingerprints_ids.extend([(str(fingerprint['_id']), info['sub_trace_i']) for info in fingerprints_info])

        DatabaseHandler.disconnect()

        """Train classifiers to obtain insights"""
        Xdat = np.array(fingerprints)
        Xdat = Xdat[:, :-5]
        Xdat = np.delete(Xdat, indexes_to_remove, axis=-1)
        ydat = np.array(labels)
        fingerprints_ids = np.array(fingerprints_ids)
        conv_dict = dict(zip(range(len(categories)), list(categories)))
        ydat = np.array([conv_dict[i] for i in ydat])
        mean_dcrs = np.array(mean_dcrs).astype(float)

        print("Computing confusion matrix")
        selected_1 = (ydat == categories[0]) | (ydat == categories[1])
        X_train, X_test, y_train, y_test = train_test_split(
            Xdat[selected_1], ydat[selected_1], test_size=0.2, random_state=42
        )
        #rus = RandomUnderSampler(replacement=False, random_state=42)
        #X_train, y_train = rus.fit_resample(X_train, y_train)
    
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        clf = OneClassSVM(nu=0.001,gamma='auto')
        clf = clf.fit(X_train)
        y_pred_isolated = clf.predict(X_test)

        selected_2 = (ydat == categories[2]) | (ydat == categories[3])
        selected_dcrs = mean_dcrs[selected_2]
        fingerprints_ids = fingerprints_ids[selected_2]
        new_ydat = ydat[selected_2]
        new_ydat[new_ydat == categories[2]] = categories[0]
        new_ydat[new_ydat == categories[3]] = categories[1]

        Xdat[selected_2] = scaler.transform(Xdat[selected_2])

        y_pred_simultaneous = clf.predict(Xdat[selected_2])
        y_pred_simultaneous = np.where(y_pred_simultaneous == -1, 1, 0)

        classification_result = defaultdict(list)
        for (fing_id,sub_trace_id), classification in zip(fingerprints_ids, y_pred_simultaneous):
            classification_result[fing_id].append({'sub_trace_i': sub_trace_id, 'classification': 'iso' if 0 == classification else 'not_iso'})

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

# %%


