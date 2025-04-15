from .Fingerprint_feat_gen import GetStates, Time_in

from .Feature import Feature


class TimeInEachState(Feature):
    def __init__(self, model):
        self.__model = model

    @property
    def names(self):
        return [r'$T_{0}$', r'$T_{1}$', r'$T_{2}$', r'$T_{3}$']

    def calculate(self, trajectory):
        states, _ = GetStates(trajectory.displacements(), self.__model)
        t0, t1, t2, t3 = Time_in(states)
        return [t0,t1,t2,t3]
