import multiprocessing
import time
import logging

from tqdm import tqdm

from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from traj_fingerprint.Features import get_trajectory_fingerprint

USE_POOL = True

def calculate_msd_parameters(traj, max_t=0.050):
    DELTA_T = 0.000132
    TIME_START = 0.000084
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

def calculate_and_save_fingerprint_for_id(arguments):
    worker_i, trace_id = arguments
    if USE_POOL:
        logger = multiprocessing.get_logger()
        start_time = time.time()
        logger.info(f"Worker {worker_i}/{trace_id} started at {start_time}")

    DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')
    trace = Trajectory.objects(id=trace_id)
    assert len(trace) == 1
    trace = trace[0]

    if 'intensity' not in trace.info:
        return
    if 'fingerprint' not in trace.info:
        return

    try:
        for state in [0,1,2,3,4]:
            trace.info['fingerprint']['dgn_state_'+str(state)] = []
            if f'eco_state_{state}' not in trace.info['analysis']:
                continue
            states = trace.info['analysis'][f'eco_state_{state}']

            for sub_trace_i, sub_trace in enumerate(trace.sub_trajectories_trajectories_from_confinement_states(custom_states=states)[1]):
                try:
                    calculate_msd_parameters(sub_trace, max_t=0.0066)
                    trace.info['fingerprint']['dgn_state_'+str(state)].append({'sub_trace_i': sub_trace_i, 'fingerprint': get_trajectory_fingerprint(sub_trace)})
                except AssertionError:
                    pass
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

    DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')
    uploaded_trajectories_ids = [str(trajectory_result['_id']) for trajectory_result in Trajectory._get_collection().find({}, {'_id':1})]
    DatabaseHandler.disconnect()

    if USE_POOL:
        pool = multiprocessing.Pool(processes=8)
        pool.map(calculate_and_save_fingerprint_for_id, list(enumerate(uploaded_trajectories_ids)))
        pool.close()
    else:
        for i, traj_id in tqdm(list(enumerate(uploaded_trajectories_ids))):
            calculate_and_save_fingerprint_for_id([i,traj_id])
