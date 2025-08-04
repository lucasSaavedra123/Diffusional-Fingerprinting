import multiprocessing
import time
import logging

from tqdm import tqdm
import numpy as np

from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from traj_fingerprint.Features import get_trajectory_fingerprint

USE_POOL = False

def calculate_msd_parameters(traj, max_t):
    DELTA_T = 0.010
    TIME_START = 0.010
    MAX_T = max_t

    t_vec, msd, d, betha, precision, goodness_of_fit = traj.temporal_average_mean_squared_displacement(
        log_log_fit_limit=MAX_T,
        limit_type='time',
        bin_width=DELTA_T,
        time_start=TIME_START,
        with_corrections=True
    )

    msd = msd[t_vec < MAX_T]
    t_vec = t_vec[t_vec < MAX_T]

    traj.info['t_vec'] = t_vec.tolist()
    traj.info['msd'] = msd.tolist()
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

def calculate_and_save_fingerprint_for_id(arguments):
    worker_i, trace_id = arguments
    if USE_POOL:
        logger = multiprocessing.get_logger()
        start_time = time.time()
        logger.info(f"Worker {worker_i}/{trace_id} started at {start_time}")

    DatabaseHandler.connect_over_network(None, None, 'localhost', 'STORM_DATA')
    trace = Trajectory.objects(id=trace_id)
    assert len(trace) == 1
    trace = trace[0]
    try:
        if 'clustered_state' in trace.info['analysis']:
            clustered_steps = np.sum(trace.info['analysis']['clustered_state'])
            clustered = clustered_steps > (trace.length//2)
            fingerprint_label = 'clustered' if clustered else 'not_clustered'
            calculate_msd_parameters(trace, 0.50)
            trace.info[f'fingerprint_{fingerprint_label}'] = get_trajectory_fingerprint(trace)
            trace.save()
    except AssertionError:
        pass
    DatabaseHandler.disconnect()

    if USE_POOL:
        end_time = time.time()
        logger.info(f"Worker {worker_i}/{trace_id} finished at {end_time} (Duration: {end_time - start_time:.2f} seconds)")

if __name__ == "__main__":
    """
    With Pool, the process is 4.42 times faster. But perhaps is more inneficient
    with many trajectories.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    multiprocessing.log_to_stderr(logging.INFO)

    DatabaseHandler.connect_over_network(None, None, 'localhost', 'STORM_DATA')
    uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({}, {'_id':1})]
    DatabaseHandler.disconnect()

    if USE_POOL:
        pool = multiprocessing.Pool(processes=8)
        pool.map(calculate_and_save_fingerprint_for_id, list(enumerate(uploaded_trajectories_ids)))
        pool.close()
    else:
        for i, traj_id in tqdm(list(enumerate(uploaded_trajectories_ids))):
            calculate_and_save_fingerprint_for_id([i,traj_id])
