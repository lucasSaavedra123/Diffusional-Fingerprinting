import numpy as np

from .Feature import Feature


class MSDAnalysis(Feature):
    def calculate(self, trajectory):
        return [
            trajectory.info['d'],
            trajectory.info['betha'],
            trajectory.info['goodness_of_fit']
        ]
