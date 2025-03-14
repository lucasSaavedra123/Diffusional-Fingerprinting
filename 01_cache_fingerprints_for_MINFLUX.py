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
import itertools

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
    DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')
    """
    With Pool, the process is 4.42 times faster. But perhaps is more inneficient
    with many trajectories.
    """
    """
    with Pool() as pool, tqdm(total=len(traces)) as pbar:
        for result in pool.imap(pool_get_trajectory_fingerprint, traces):
            fingerprints.append(result)
            pbar.update()
            pbar.refresh()
    """
    uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({}, {'_id':1})]

    for trace_id in tqdm(uploaded_trajectories_ids):
        trace = Trajectory.objects(id=trace_id)
        assert len(trace) == 1
        trace = trace[0]

        if 'fingerprint' not in trace.info:
            try:
                calculate_msd_parameters(trace)
                trace.info['fingerprint'] = get_trajectory_fingerprint(trace)
                delete_fields = ['t_vec', 'msd', 'd', 'betha', 'precision', 'goodness_of_fit']
                for field in delete_fields:
                    del trace.info[field]
                trace.save()
            except AssertionError:
                pass

    DatabaseHandler.disconnect()
