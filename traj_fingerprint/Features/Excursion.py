import numpy as np

from .Feature import Feature


class Excursion(Feature):
    """
    Obtained from (Kowalek, 2022)
    """
    def calculate(self, trajectory):
        raw = trajectory.raw_trajectory
        maximal_excursion = np.max(trajectory.displacements())/np.linalg.norm(raw[0]-raw[-1])

        d_n = np.max([np.linalg.norm(raw[0]-raw[i]) for i in range(1, trajectory.length)])
        d_n_std = (1/(2*trajectory.length*np.mean(trajectory.get_time())))*np.sum(np.array(trajectory.displacements())**2)
        mean_maximal_excursion = d_n/np.sqrt((d_n_std**2)*trajectory.duration)

        return [
            maximal_excursion,
            mean_maximal_excursion,
        ]