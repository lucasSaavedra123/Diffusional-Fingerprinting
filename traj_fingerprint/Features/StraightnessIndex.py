import numpy as np

from .Feature import Feature


class StraightnessIndex(Feature):
    """
    Obtained from https://doi.org/10.3390/receptors4010006
    URL: https://github.com/okovtun86/diffusion_classifier_3d/blob/61ce001b7f3f9c50f1378b70bfadfb0312ba0549/gen_fingerprints_3d.py#L35C5-L37C67
    """
    def calculate(self, trajectory):
        x,y = trajectory.get_noisy_x(), trajectory.get_noisy_y()
        total_path_length = np.sum(trajectory.displacements())
        trajectory = np.array([x,y]).T
        end_to_start_distance = np.linalg.norm((trajectory-trajectory[0])[-1])
        straightness_index = end_to_start_distance / total_path_length
        return [straightness_index]
