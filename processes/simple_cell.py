from vivarium.core.process import Process, Step
from vivarium.core.engine import Engine, pf
from vivarium.plots.simulation_output import plot_simulation_output


class SimpleCell(Process):
    defaults = {
        'k_sxs': 0.02,
        'k_sx': 1,
        'k_px': 1,
        'k_dx': 0.2,
        'k_dsx': 1,
        'k_sy': 0.01,
        'k_py': 1,
        'k_dy': 0.01,
        'V_g': 1,
        'k_g': 0.05,
        'd_g': 0.1,
        'HIF_initial': 0.1,
        'Lactate_initial': 0.2,
        'GFP_initial': 0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.species_list = ['HIF', 'Lactate', 'GFP', 'oxygen']

    def initial_state(self, config=None):
        return {
            'species': {
                'HIF': self.parameters['HIF_initial'],
                'Lactate': self.parameters['Lactate_initial'],
                'GFP': self.parameters['GFP_initial'],
            }
        }
    
    def ports_schema(self):
        return {
            'species': {
                species_id: {
                    '_default': 0.0,
                    '_emit': True,
                } for species_id in self.species_list},
        }

    def next_update(self, interval, states):
        # get the states
        species = states['species']

        # run the simulation
        dHIF = (self.parameters['k_sxs'] + self.parameters['k_sx'] * (species['HIF']**2) /
                (self.parameters['k_px'] + species['HIF']**2) - self.parameters['k_dx'] * species['HIF'] -
                self.parameters['k_dsx'] * species['HIF'] * species['Lactate'])
        dLactate = (self.parameters['k_sy'] * species['HIF']**2 / (self.parameters['k_py'] + species['HIF']**2) -
                    self.parameters['k_dy'] * species['Lactate'])
        dGFP = (self.parameters['V_g'] * species['HIF']**3 / (self.parameters['k_g'] + species['HIF']**3) -
                self.parameters['d_g'] * species['GFP'])

        # retrieve the results
        return {
            'species': {
                'HIF': dHIF * interval,
                'Lactate': dLactate * interval,
                'GFP': dGFP * interval
            }
        }


def test_cell():
    total_time = 1000

    # create the process
    parameters = {'timestep': 0.1}
    cell = SimpleCell(parameters=parameters)

    # put it in a simulation
    sim = Engine(
        initial_state=cell.initial_state(),
        processes={'cell': cell},
        topology={'cell': {'species': ('species',)}}
    )

    # run simulation
    sim.update(total_time)

    # get the results
    data = sim.emitter.get_timeseries()

    # # print results
    # print(pf(data))

    # plot results
    settings = {}
    plot_simulation_output(
        data,
        settings=settings,
        out_dir='out',
        filename='results'
    )

if __name__ == '__main__':
    test_cell()
