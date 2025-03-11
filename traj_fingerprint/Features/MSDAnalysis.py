import numpy as np

from .Feature import Feature


class MSDAnalysis(Feature):
    def calculate(self, trajectory):
        DELTA_T = 0.000132
        TIME_START = 0.000084
        MAX_T = 0.050

        _, _, d, betha, precision, goodness_of_fit = trajectory.temporal_average_mean_squared_displacement(
            log_log_fit_limit=MAX_T,
            limit_type='time',
            bin_width=DELTA_T,
            time_start=TIME_START,
            with_corrections=True
        )

        return [d, betha, goodness_of_fit]
