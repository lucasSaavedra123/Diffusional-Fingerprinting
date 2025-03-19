import numpy as np

from .Feature import Feature


class RadiusOfGyration(Feature):
    """
    Obtained from https://doi.org/10.3390/receptors4010006
    However, I found other definition by (Wagner, 2017)
    and (Kowalek, 2022), (Kowalek, 2019) which works 
    with eigenvalues (also called Asymmetry; perhaps 
    It's another thing...).
    Check which definition to use.
    """
    def calculate(self, trajectory):
        x,y = trajectory.get_noisy_x(), trajectory.get_noisy_y()
        x_com, y_com = np.mean(x), np.mean(y)
        return [np.sqrt(np.mean(((x-x_com)**2)+((y-y_com)**2)))]
