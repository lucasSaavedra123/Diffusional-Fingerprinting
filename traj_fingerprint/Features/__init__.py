import os

from pomegranate import *
import numpy as np

from .Feature import Feature
from .MSDAnalysis import MSDAnalysis
from .Efficiency import Efficiency
from .FractalDimension import FractalDimension
from .Gaussianity import Gaussianity
from .Kurtosis import Kurtosis
from .MSDRatio import MSDRatio
from .Trappedness import Trappedness
from .TimeInEachState import TimeInEachState
from .Lifetime import Lifetime
from .TrajectoryLength import TrajectoryLength
from .MeanDisplacements import MeanDisplacements
from .MeanMSD import MeanMSD

print("Loading HMM model...")
file = open(os.path.join(__file__,'..','..','..','HMMjson'), "r")
json_s = ""
for line in file:
    json_s += line
hmm_model = HiddenMarkovModel.from_json(json_s)
print("Loaded model:")
print(hmm_model)

features = Feature.__subclasses__()
features.remove(Lifetime)
features.remove(TimeInEachState)

IMPLEMENTED_FEATURES = [feature() for feature in features]
IMPLEMENTED_FEATURES.append(Lifetime(hmm_model))
IMPLEMENTED_FEATURES.append(TimeInEachState(hmm_model))
IMPLEMENTED_FEATURES = tuple(IMPLEMENTED_FEATURES)

def get_trajectory_fingerprint(traj):
    fingerprint = []
    for feature in IMPLEMENTED_FEATURES:
        fingerprint.extend(feature.calculate(traj))
    return np.array(fingerprint)
