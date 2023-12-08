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

        # TODO transport
        'Lactate_export': 0.0,
        'Lactate_import': 1.0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.internal_species_list = ['HIF', 'Lactate', 'GFP', 'oxygen']
        self.external_species_list = ['Lactate_ext']

    def initial_state(self, config=None):
        return {
            'internal_species': {
                'HIF': self.parameters['HIF_initial'],
                'Lactate': self.parameters['Lactate_initial'],
                'GFP': self.parameters['GFP_initial'],
            }
        }
    
    def ports_schema(self):
        return {
            'internal_species': {
                species_id: {
                    '_default': 0.0,
                    '_emit': True,
                } for species_id in self.internal_species_list
            },
            'external_species': {}
        }

    def next_update(self, interval, states):
        # get the states
        internal_species = states['internal_species']

        """
        dHIF = ksxs + ksx*(HIF^2)/(kpx + HIF^2) - kdx*HIF - kdsx*HIF*Lactate
        dLactate = ksy*HIF^2/(kpy + HIF^2) - kdy*Lactate
        dGFP = Vg*HIF^3/(kg+HIF^3) - dg*GFP
        """


        # run the simulation
        dHIF = (self.parameters['k_sxs'] + self.parameters['k_sx'] * (internal_species['HIF']**2) /
                (self.parameters['k_px'] + internal_species['HIF']**2) - self.parameters['k_dx'] * internal_species['HIF'] -
                self.parameters['k_dsx'] * internal_species['HIF'] * internal_species['Lactate'])
        dLactate = (self.parameters['k_sy'] * internal_species['HIF']**2 / (self.parameters['k_py'] + internal_species['HIF']**2) -
                    self.parameters['k_dy'] * internal_species['Lactate'])
        dGFP = (self.parameters['V_g'] * internal_species['HIF']**3 / (self.parameters['k_g'] + internal_species['HIF']**3) -
                self.parameters['d_g'] * internal_species['GFP'])

        # retrieve the results
        return {
            'internal_species': {
                'HIF': dHIF * interval,
                'Lactate': dLactate * interval,
                'GFP': dGFP * interval
            }
        }


def test_cell():
    total_time = 1000

    # create the process
    parameters = {
        'timestep': 0.1,
        'k_sx': 1.0,  # default is 1. bifurcation ~ 0.7, 1.4
        'Lactate_initial': 0.001 # default 0.2
    }
    cell = SimpleCell(parameters=parameters)

    # put it in a simulation
    sim = Engine(
        initial_state=cell.initial_state(),
        processes={'cell': cell},
        topology={'cell': {'internal_species': ('internal_species',)}}
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
