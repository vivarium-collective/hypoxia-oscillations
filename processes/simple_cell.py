from vivarium.core.process import Process, Step


class Cell(Process):
    defaults = {}

    def __init__(self, parameters):
        super().__init__(parameters)

    def ports_schema(self):
        return {
            'uptake': {},
            'secretion': {}
        }

    def next_update(self, timestep, states):
        # set the states

        # run the simulation

        # retrieve the results

        return {}


