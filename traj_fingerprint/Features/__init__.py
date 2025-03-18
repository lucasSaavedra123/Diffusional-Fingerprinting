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
from .Displacements import Displacements
from .MeanMSD import MeanMSD
from .Velocity import Velocity
from .RadialDistances import RadialDistances
from .RadiusOfGyration import RadiusOfGyration
from .StraightnessIndex import StraightnessIndex
from .Directionality import Directionality


file = open(os.path.join(__file__,'..','..','..','HMMjson'), "r")
json_s = ""
for line in file:
    json_s += line
hmm_model = HiddenMarkovModel.from_json(json_s)

features = Feature.__subclasses__()
features.remove(Lifetime)
features.remove(TimeInEachState)

FEATURES = [feature() for feature in features]
FEATURES.append(Lifetime(hmm_model))
FEATURES.append(TimeInEachState(hmm_model))
FEATURES = tuple(FEATURES)

def get_trajectory_fingerprint(traj):
    fingerprint = []
    for feature in FEATURES:
        fingerprint.extend(feature.calculate(traj))
    return fingerprint
