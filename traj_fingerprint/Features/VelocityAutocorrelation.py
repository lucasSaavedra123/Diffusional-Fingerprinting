import numpy as np

from .Feature import Feature


def empirical_velocitiy_autocorrelation(n, raw_t):
    dots = []
    for i in range(0, len(raw_t)-2):
        t_i_1 = raw_t[i+1]
        t_i_1_n = raw_t[i+1+n]
        t_i = raw_t[i]
        t_i_n = raw_t[i+n]

        dots.append(np.linalg.norm(t_i_1_n-t_i_n) * np.linalg.norm(t_i_1-t_i))

    return np.sum(dots)/(len(raw_t)-1)

class VelocityAutocorrelation(Feature):
    """
    Obtained from (Kowalek, 2022)
    """
    def calculate(self, trajectory):
        return [
            empirical_velocitiy_autocorrelation(1,trajectory.raw_trajectory),
            empirical_velocitiy_autocorrelation(2,trajectory.raw_trajectory),
        ]