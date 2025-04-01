import numpy as np

from .Feature import Feature


class Intensity(Feature):
    @property
    def names(self):
        return ['meanDCR', 'meanEFO', 'meanECO']

    def calculate(self, trajectory):
        info = trajectory.info
        t = trajectory.get_time()

        if 'dcr' in info and 'intensity' in info:
            dcr = np.mean(info['dcr'])
            efo = np.mean(info['intensity'])
            eco = np.mean(np.diff(t) * info['intensity'][:-1])
            return [dcr,efo,eco]
        else:
            #If not available, just None
            return [None,None,None]
