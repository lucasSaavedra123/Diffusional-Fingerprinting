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

color = {
    1:'red',
    4:'blue',
    8:'green',
    16:'purple',
    32:'orange'
}

DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')

for query_string in queries:
    query = list(Trajectory._get_collection().find(queries[query_string], {'x':1,'y':1, f'info.analysis.predictions_on_each_100_segments':1}))

    plt.rcParams['font.size'] = '16'
    fig, ax = plt.subplots(1,2,figsize=(15,5))

    for steps_lag in [1,4,8,16,32]:
        angles = {False: [], True: []}

        for q_i, q in enumerate(query):
            t = np.array([q['x'],q['y']]).T

            if 'predictions_on_each_100_segments' not in q.get('info', {}).get('analysis', {}):
                continue

            q = q['info']['analysis']['predictions_on_each_100_segments']

            for i in range(len(q)):
                angles[q[i][0] == q[i][1]].extend(turning_angles(len(t), t[:,0], t[:,1], steps_lag=steps_lag))

        for boolean in angles:
            frequency, bin_edges = custom_histogram(np.array(angles[boolean]), 0, 180, 20)
            probability = frequency / np.sum(frequency)
            x_mid = []
            for i in range(len(probability)):
                x_mid.append((bin_edges[i+1] + bin_edges[i]) / 2)
            delta_x = bin_edges[1] - bin_edges[0]
            f_x = probability / delta_x
            ax[int(boolean)].plot(x_mid,f_x, color=color[steps_lag], label=f'Step lag={steps_lag}', linewidth=5)

    ax[0].set_title('Misclassified segments')
    ax[0].set_xlabel('Turning angle')
    ax[0].set_ylabel('Probability Density Function')
    ax[0].set_ylim(0.000,0.012)
    ax[0].set_xlim(0,180)
    ax[0].legend()

    ax[1].set_title('Correctly classified segments')
    ax[1].set_xlabel('Turning angle')
    ax[1].set_ylim(0.000,0.012)
    ax[1].set_xlim(0,180)

    plt.savefig(os.path.join(PROJECT_PATH, 'Graphics/Matplotlib', f"TURNING_ANGLE_PDF_{query_string}.svg"))

DatabaseHandler.disconnect()
