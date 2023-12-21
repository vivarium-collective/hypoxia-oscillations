from vivarium.core.process import Process, Step
from vivarium.core.engine import Engine, pf
from vivarium.plots.simulation_output import plot_simulation_output


class SimpleCell(Process):
    defaults = {
        'k_HIF_production_basal': 0.02,  # k_sxs
        'k_HIF_production_max': 1,  # k_sx
        'k_HIF_pos_feedback': 1,  # k_px
        'k_HIF_deg_basal': 0.2,  # k_dx
        'k_HIF_deg_lactate': 1,  # k_dsx
        'k_Lactate_production': 0.01,  # k_sy
        'k_Lactate_production_reg': 1,  # k_py
        'k_Lactate_deg_basal': 0.01,  # k_dy
        'k_GFP_production_constantFP_production': 1,  # V_g
        'k_GFP_production_constant': 0.05,  # k_g
        'k_GFP_deg': 0.1,  # d_g

        # initial states
        'HIF_initial': 0.1,  #
        'Lactate_initial': 0.2,  #
        'external_Lactate_initial': 0.1,
        'GFP_initial': 0.0,  #

        # TODO transport
        'k_MCT1': 1E-3,  # Lactate import
        'k_MCT4': 1E-3,  # Lactate export

        # oxygen consumption
        'k_O2_consumption': 1.0,
        'external_oxygen_initial': 1.0,

    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.internal_species_list = ['HIF', 'Lactate', 'GFP', 'oxygen']
        self.external_species_list = ['Lactate', 'oxygen']

    def initial_state(self, config=None):
        return {
            'internal_species': {
                'HIF': config.get('HIF_initial') or self.parameters['HIF_initial'],
                'Lactate': config.get('Lactate_initial') or self.parameters['Lactate_initial'],
                'GFP': config.get('GFP_initial') or self.parameters['GFP_initial'],
            },
            'external_species': {
                'Lactate': config.get('external_Lactate_initial') or self.parameters['external_Lactate_initial'],
                'oxygen': config.get('external_oxygen_initial') or self.parameters['external_oxygen_initial'],
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
            'external_species': {
                species_id: {
                    '_default': 0.0,
                    '_emit': True,
                } for species_id in self.external_species_list
            }
        }

    def next_update(self, interval, states):
        # get the states
        internal_species = states['internal_species']
        external_species = states['external_species']

        """
        dHIF = ksxs + ksx*(HIF^2)/(kpx + HIF^2) - kdx*HIF - kdsx*HIF*Lactate
        dLactate = ksy*HIF^2/(kpy + HIF^2) - kdy*Lactate
        dGFP = Vg*HIF^3/(kg+HIF^3) - dg*GFP
        """

        # run the simulation
        dHIF = ((self.parameters['k_HIF_production_basal'] + self.parameters['k_HIF_production_max'] * (internal_species['HIF']**2) /
                (self.parameters['k_HIF_pos_feedback'] + internal_species['HIF']**2) - self.parameters['k_HIF_deg_basal'] * internal_species['HIF'] * external_species['oxygen'] -
                self.parameters['k_HIF_deg_lactate'] * internal_species['HIF'] * internal_species['Lactate']))
        dLactate = (self.parameters['k_Lactate_production'] * internal_species['HIF']**2 / (self.parameters['k_Lactate_production_reg'] + internal_species['HIF']**2) -
                    self.parameters['k_Lactate_deg_basal'] * internal_species['Lactate']) + (
                    self.parameters['k_MCT1'] * external_species['Lactate'] - self.parameters['k_MCT4'] * internal_species['Lactate'])  # Get improved function fo Lactate transport
        dGFP = (self.parameters['k_GFP_production_constantFP_production'] * internal_species['HIF']**3 / (self.parameters['k_GFP_production_constant'] + internal_species['HIF']**3) -
                self.parameters['k_GFP_deg'] * internal_species['GFP'])
        # dO2 = ()

        # retrieve the results
        return {
            'internal_species': {
                'HIF': dHIF * interval,
                'Lactate': dLactate * interval,
                'GFP': dGFP * interval
            },
            'external_species': {
                # TODO -- we need to update external Oxygen,
            }
        }


def test_cell():
    total_time = 1000

    # create the process
    parameters = {
        'timestep': 0.1,
        'k_HIF_production_max': 0.9,  # default is 1. bifurcation ~ 0.7, 1.4
        'Lactate_initial': 0.001,  # default 0.2
        'external_Lactate_initial': 0.1,
        'external_oxygen_initial': 1.1,
    }
    cell = SimpleCell(parameters=parameters)

    # put it in a simulation
    sim = Engine(
        initial_state=cell.initial_state({}),
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
