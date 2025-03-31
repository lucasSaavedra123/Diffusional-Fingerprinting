import os

from pomegranate import *


current_directory = os.path.split(__file__)[0]
features_files = os.listdir(current_directory)
features_files.remove('__pycache__')
features_files.remove('__init__.py')
features_files.remove('Fingerprint_feat_gen.py')

for feature_file in features_files:
    feature_file = feature_file.split('.')[0]
    exec(f"from .{feature_file} import {feature_file}")

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

def get_feature_names():
    names = []
    for feature in FEATURES:
        names.extend(feature.names)
    return names
