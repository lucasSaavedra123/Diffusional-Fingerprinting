from .Fingerprint_feat_gen import Gaussianity as rawGaussianity

from .Feature import Feature


class Gaussianity(Feature):
    def calculate(self, trajectory):
        DELTA_T = 0.000132
        TIME_START = 0.000084
        MAX_T = 0.050

        _, msd, _, _, _, _ = trajectory.temporal_average_mean_squared_displacement(
            log_log_fit_limit=MAX_T,
            limit_type='time',
            bin_width=DELTA_T,
            time_start=TIME_START,
            with_corrections=True
        )

        return [rawGaussianity(
            trajectory.get_noisy_x(),
            trajectory.get_noisy_y(),
            msd
        )]
