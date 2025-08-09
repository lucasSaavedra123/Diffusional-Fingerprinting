from collections import Counter
import numpy as np
from tqdm import tqdm
from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from MLGeneral import ML
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

if __name__ == "__main__":
    DatabaseHandler.connect_over_network(None, None, 'localhost', 'STORM_DATA')

    fingerprints = []
    labels = []

    """
    clustered_fingerprints = Trajectory._get_collection().find({}, {f'info.fingerprint_clustered':1})
    not_clustered_fingerprints = Trajectory._get_collection().find({}, {f'info.fingerprint_unclustered':1})

    for fingerprint in tqdm(clustered_fingerprints):
        if 'fingerprint_clustered' in fingerprint['info'] and 0 < len(fingerprint['info']['fingerprint_clustered']):
            fingerprints.extend(fingerprint['info']['fingerprint_clustered'])
            labels += len(fingerprint['info']['fingerprint_clustered'])*['clustered']

    for fingerprint in tqdm(not_clustered_fingerprints):
        if 'fingerprint_unclustered' in fingerprint['info'] and 0 < len(fingerprint['info']['fingerprint_unclustered']):
            fingerprints.extend(fingerprint['info']['fingerprint_unclustered'])
            labels += len(fingerprint['info']['fingerprint_unclustered'])*['unclustered']
    """

    raw_fingerprints = Trajectory._get_collection().find(
        {'info.immobile':False},
        {f'info.fingerprint_full':1,'info.analysis.clustered_state':1}
    )

    for info in raw_fingerprints:
        if 'analysis' in info['info'] and 'fingerprint_full' in info['info'] and 'clustered_state' in info['info']['analysis']:
            fingerprints.append(info['info']['fingerprint_full'])
            label = (np.sum(info['info']['analysis']['clustered_state']) > (len(info['info']['analysis']['clustered_state'])//2)).astype(int)
            labels.append('clustered' if label == 1 else 'unclustered')

    DatabaseHandler.disconnect()

    fingerprints = np.array(fingerprints)
    fingerprints = fingerprints[:, :-5]
    fingerprints = fingerprints.astype(float)

    kept_fingerprints = ~np.isnan(fingerprints).any(axis=0)

    fingerprints = fingerprints[:, kept_fingerprints]

    labels = np.array(labels)

    print(fingerprints.shape, labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        fingerprints, labels, test_size=0.10, shuffle=True, random_state=42
    )

    rus = RandomUnderSampler(replacement=False, random_state=42)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    learn = ML(X_train, y_train)
    learn.Train(algorithm='Boost', plot=False)
    y_pred = learn.Predict(ML(X_test, y_test, center=False))[0]
    y_pred = [learn.to_string[i] for i in y_pred]

    cm = confusion_matrix(y_test, y_pred)
    cm = np.round(cm/(np.sum(cm,axis=1).reshape(2,1)),2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Unclustered', 'Clustered'])
    disp.plot()
    plt.show()

    DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')

    queries = {
        'CDx':{'info.dataset':'CDx'},
        'CF®680R-BTX':{'info.dataset':'BTX680R'},
        'BTX640R':{'info.dataset':'Control'},
        'CF®680R-BTX(+fPEG-Chol)':{'info.dataset':'Cholesterol and btx', 'info.classified_experimental_condition':'BTX680R'},
    }

    for category_id, category in enumerate(queries):
        fingerprints = []
        queries[category].update({'info.immobile': False})
        query_fingerprints = Trajectory._get_collection().find(queries[category], {f'info.fingerprint.undersampled':1})
        for fingerprint in tqdm(query_fingerprints):
            if 'fingerprint' in fingerprint['info'] and 'undersampled' in fingerprint['info']['fingerprint']:
                fingerprints = []
                fingerprints.append(fingerprint['info']['fingerprint']['undersampled'])
                fingerprints = np.array(fingerprints)
                fingerprints = fingerprints[:, :-5]
                fingerprints = fingerprints[:, kept_fingerprints]
                fingerprints = fingerprints.astype(float)

                prediction = learn.Predict(ML(fingerprints, np.zeros((len(fingerprints))), center=False))[0]
                prediction = [learn.to_string[i] for i in prediction][0]

                traj = Trajectory.objects(id=fingerprint['_id'])[0]
                traj.info['analysis']['storm_based_analysis_cluster'] = prediction
                traj.save()

    DatabaseHandler.disconnect()