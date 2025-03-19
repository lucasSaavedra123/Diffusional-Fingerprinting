import numpy as np
from scipy.stats import kurtosis, skew

from .Feature import Feature


class Displacements(Feature):
    """
    Obtained from https://doi.org/10.21203/rs.3.rs-3716053/v1
    , (Pinholt, 2019), and (Wimmenauer, 2023)
    """
    def calculate(self, trajectory):
        displacements = trajectory.displacements()
        return [
            np.mean(displacements),
            np.std(displacements),
            kurtosis(displacements),
            skew(displacements),
            np.min(displacements),
            np.max(displacements),
            np.std(displacements)/np.mean(displacements)
        ]
