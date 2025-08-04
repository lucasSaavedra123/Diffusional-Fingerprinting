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

if __name__ == "__main__":
    DatabaseHandler.connect_over_network(None, None, 'localhost', 'STORM_DATA')

    fingerprints = []
    labels = []

    clustered_fingerprints = Trajectory._get_collection().find({}, {f'info.fingerprint_clustered':1})
    not_clustered_fingerprints = Trajectory._get_collection().find({}, {f'info.fingerprint_not_clustered':1})

    for fingerprint in tqdm(clustered_fingerprints):
        if 'fingerprint_clustered' in fingerprint['info']:
            fingerprints.append(fingerprint['info']['fingerprint_clustered'])
            labels.append(1)

    for fingerprint in tqdm(not_clustered_fingerprints):
        if 'fingerprint_not_clustered' in fingerprint['info']:
            fingerprints.append(fingerprint['info']['fingerprint_not_clustered'])
            labels.append(0)

    DatabaseHandler.disconnect()

    fingerprints = np.array(fingerprints)
    fingerprints = fingerprints[:, :-5]
    fingerprints = fingerprints.astype(float)
    fingerprints = fingerprints[:, ~np.isnan(fingerprints).any(axis=0)]

    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        fingerprints, labels, test_size=0.2, shuffle=True
    )

    rus = RandomUnderSampler(replacement=False, random_state=42)
    X_train, y_train = rus.fit_resample(X_train, y_train)

    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=1000, verbose=0)
    clf.fit(X_train, y_train)

    pred = 1 - clf.predict(X_test)

    cm = confusion_matrix(y_test, pred, labels=clf.classes_)
    cm = np.round(cm/(np.sum(cm,axis=1).reshape(2,1)),2)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()
