import matplotlib.pyplot as plt

import os
import numpy as np
# import multiprocess as mp
from Trajectory import Trajectory
from DatabaseHandler import DatabaseHandler
from CONSTANTS import PROJECT_PATH


queries = {
    'CF®680R-BTX(+fPEG-Chol)':{'info.dataset':'Cholesterol and btx', 'info.classified_experimental_condition':'BTX680R'},
    'fPEG-Chol(+CF®680R-BTX)':{'info.dataset':'Cholesterol and btx', 'info.classified_experimental_condition':'fPEG-Chol'},
}

DatabaseHandler.connect_over_network(None, None, 'localhost', 'MINFLUX_DATA')

for query_string in queries:
    query = list(Trajectory._get_collection().find(queries[query_string], {'x':1,'y':1, f'info.analysis.predictions_on_each_100_segments':1}))

    for q_i, q in enumerate(query):
        t = np.array([q['x'],q['y']]).T

        if 'predictions_on_each_100_segments' not in q.get('info', {}).get('analysis', {}):
            continue

        q = q['info']['analysis']['predictions_on_each_100_segments']

        for i in range(len(q)):
            color='green' if q[i][0] == q[i][1] else 'red'
            plt.plot(t[i*100:(i+1)*100,0], t[i*100:(i+1)*100,1], color=color)

        plt.savefig(os.path.join(PROJECT_PATH, 'Graphics/Matplotlib/TRAJS', f"TRAJ_{query_string}_{q_i}.svg"))
        plt.clf()
