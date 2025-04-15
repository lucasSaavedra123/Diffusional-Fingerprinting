import matplotlib.pyplot as plt

import os
import numpy as np
# import multiprocess as mp
from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from multiprocessing import Pool
from CONSTANTS import PROJECT_PATH
from Trajectory import turning_angles


def custom_histogram(data, starting_x, final_x, x_step):
  bin_edges = [starting_x]
  frequency = []

  current_x = starting_x

  while current_x < final_x:
    left = data > current_x
    right = data < (current_x + x_step)

    frequency.append(np.sum(np.logical_and(left,right).astype(int)))
    current_x += x_step

    bin_edges.append(current_x)

  return frequency, bin_edges

queries = {
    #'CF®680R-BTX':{'info.dataset':'BTX680R'},
    #'BTX640R':{'info.dataset':'Control'},
    #'fPEG-Chol':{'info.dataset':'CholesterolPEGKK114'},
    'CF®680R-BTX(+fPEG-Chol)':{'info.dataset':'Cholesterol and btx', 'info.classified_experimental_condition':'BTX680R'},
    'fPEG-Chol(+CF®680R-BTX)':{'info.dataset':'Cholesterol and btx', 'info.classified_experimental_condition':'fPEG-Chol'},
}

DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')

plt.rcParams['font.size'] = '16'
fig, ax = plt.subplots(1,2,figsize=(12,5))

for query_i, query_string in enumerate(queries):
    query = list(Trajectory._get_collection().find(queries[query_string], {'x':1,'y':1, f'info.analysis.predictions_on_each_100_segments':1}))

    x_step_lag = []
    y_mean_turning_angle = {False:[], True:[]}

    for steps_lag in np.arange(1,50,1):
        angles = {False: [], True: []}

        for q in query:
            t = np.array([q['x'],q['y']]).T

            if 'predictions_on_each_100_segments' not in q.get('info', {}).get('analysis', {}):
                continue

            q = q['info']['analysis']['predictions_on_each_100_segments']

            for i in range(len(q)):
                angles[q[i][0] == q[i][1]].extend(turning_angles(len(t), t[:,0], t[:,1], steps_lag=steps_lag))

        x_step_lag.append(steps_lag)
        y_mean_turning_angle[False].append(np.mean(angles[False]))
        y_mean_turning_angle[True].append(np.mean(angles[True]))

    ax[query_i].plot(x_step_lag,y_mean_turning_angle[False], label='Misclassified segments', color='red', linewidth=3)
    ax[query_i].plot(x_step_lag,y_mean_turning_angle[True], label='Correctly classified segments', color='green', linewidth=3)
    ax[query_i].set_title(query_string)
    ax[query_i].set_xlabel('Step lag')
    ax[query_i].set_ylabel('Average Turning angle')
    ax[query_i].set_xlim(1,50)
    ax[query_i].set_ylim(0,180)
    ax[query_i].set_yticks([0,45,90,135,180])
    ax[query_i].axhline(180/2, linestyle='--', linewidth=3, color='black')

ax[0].legend()
plt.savefig(os.path.join(PROJECT_PATH, 'Graphics/Matplotlib', f"TURNING_ANGLE_STEP_LAG_VS_MEAN_ANGLE.svg"))

DatabaseHandler.disconnect()
