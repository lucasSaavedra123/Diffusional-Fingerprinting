import numpy as np

from .Feature import Feature


class MSDAnalysis(Feature):
    @property
    def names(self):
        return [r'$D_{coeff}$', r'$\beta$', r'$GOF$']

    """
    In almost all papers for Fingerprinting this
    analysis is included. Check later for precise
    references.
    """
    def calculate(self, trajectory):
        return [
            trajectory.info['d'],
            trajectory.info['betha'],
            trajectory.info['goodness_of_fit']
        ]
