from .Fingerprint_feat_gen import GetStates, Time_in

from .Feature import Feature


class TimeInEachState(Feature):
    def __init__(self, model):
        self.__model = model

    @property
    def names(self):
        return ['T0', 'T1', 'T2', 'T3']

    def calculate(self, trajectory):
        states, _ = GetStates(trajectory.displacements(), self.__model)
        t0, t1, t2, t3 = Time_in(states)
        return [t0,t1,t2,t3]
