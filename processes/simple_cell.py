from vivarium.core.process import Process, Step
from vivarium.core.engine import Engine, pf
from vivarium.plots.simulation_output import plot_simulation_output
# from processes.local_field import MOLECULAR_WEIGHTS, AVOGADRO
# import numpy as np


class SimpleCell(Process):
    defaults = {
        'k_HIF_production_basal': 0.02,  # k_sxs
        'k_HIF_production_max': 0.9,  # k_sx
        'k_HIF_pos_feedback': 1,  # k_px
        'k_HIF_deg_basal': 0.2,  # k_dx
        'k_HIF_deg_lactate': 1,  # k_dsx
        'k_lactate_production': 0.01,  # k_sy
        'k_lactate_production_reg': 1,  # k_py
        'k_lactate_deg_basal': 0.01,  # k_dy
        'k_GFP_production_constantFP_production': 1,  # V_g
        'k_GFP_production_constant': 0.05,  # k_g
        'k_GFP_deg': 0.1,  # d_g

        # initial states
        'HIF_initial': 0.1,  #
        'lactate_initial': 0.001,  #
        'external_lactate_initial': 0.1,
        'GFP_initial': 0.0,  #

        # TODO transport
        'k_MCT1': 1E-3,  # lactate import
        'k_MCT4': 1E-3,  # lactate export

        # oxygen consumption
        'k_O2_consumption': 1.0,
        'external_oxygen_initial': 1.1,

        # oxygen exchange
        'kmax_o2_deg': 1e-1,
        'HIF_threshold': 2.5,
        'hill_coeff_o2_deg': 10,
        'k_min_o2_deg': 1e-2
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.internal_species_list = ['HIF', 'lactate', 'GFP', 'oxygen']
        self.external_species_list = ['lactate', 'oxygen']

        self.conc_conversion = {
            'oxygen': 1.0,
            'lactate': 10.0,
        }  # TODO set conversion from conc to counts

    def initial_state(self, config=None):
        return {
            'internal_species': {
                'HIF': config.get('HIF_initial') or self.parameters['HIF_initial'],
                'lactate': config.get('lactate_initial') or self.parameters['lactate_initial'],
                'GFP': config.get('GFP_initial') or self.parameters['GFP_initial'],
            },
            'boundary': {
                'external': {
                    'lactate': config.get('external_lactate_initial') or self.parameters['external_lactate_initial'],
                    'oxygen': config.get('external_oxygen_initial') or self.parameters['external_oxygen_initial'],
                }
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
            'boundary': {
                # external is how the cell sees its surroundings
                'external': {
                    species_id: {
                        '_default': 0.0,
                        '_emit': True,
                    } for species_id in self.external_species_list
                },
                # exchange is how the cell updates its surroundings
                'exchange': {
                    species_id: {
                        '_default': 0.0,
                        '_emit': True,
                    } for species_id in self.external_species_list
                }
            }
        }

    def next_update(self, interval, states):
        """
        dHIF = ksxs + ksx*(HIF^2)/(kpx + HIF^2) - kdx*HIF - kdsx*HIF*lactate
        dlactate = ksy*HIF^2/(kpy + HIF^2) - kdy*lactate
        dGFP = Vg*HIF^3/(kg+HIF^3) - dg*GFP
        """

        # get the variables
        hif_in = states['internal_species']['HIF']
        oxygen_ex = states['boundary']['external']['oxygen']
        lactate_in = states['internal_species']['lactate']
        lactate_ex = states['boundary']['external']['lactate']
        gfp_in = states['internal_species']['GFP']

        # Calculate the rate of change of HIF
        hif_production = (
                self.parameters['k_HIF_production_basal'] +
                self.parameters['k_HIF_production_max'] * hif_in ** 2 /
                (self.parameters['k_HIF_pos_feedback'] + hif_in ** 2)
        )
        hif_degradation = (
                self.parameters['k_HIF_deg_basal'] * hif_in * oxygen_ex +
                self.parameters['k_HIF_deg_lactate'] * hif_in * lactate_in
        )
        dHIF = hif_production - hif_degradation

        # Calculate the rate of change of Lactate
        lactate_production = (
                self.parameters['k_lactate_production'] * hif_in ** 2 /
                (self.parameters['k_lactate_production_reg'] + hif_in ** 2)
        )
        lactate_degradation = self.parameters['k_lactate_deg_basal'] * lactate_in
        lactate_transport = (  # TODO: improve function for lactate transport
                self.parameters['k_MCT1'] * lactate_ex -
                self.parameters['k_MCT4'] * lactate_in
        )
        dLactate = lactate_production - lactate_degradation + lactate_transport

        # Calculate the rate of change of GFP
        gfp_production = (
                self.parameters['k_GFP_production_constantFP_production'] * hif_in ** 3 /
                (self.parameters['k_GFP_production_constant'] + hif_in ** 3)
        )
        gfp_degradation = self.parameters['k_GFP_deg'] * gfp_in
        dGFP = gfp_production - gfp_degradation

        # Calculate oxygen exchange
        dO2_ext = 0
        if oxygen_ex > 0:
            dO2_ext = - self.parameters['k_min_o2_deg'] - self.parameters['kmax_o2_deg'] / (
                    (hif_in / self.parameters['HIF_threshold'])**self.parameters['hill_coeff_o2_deg'] + 1)

        # convert dO2 and lactate transport from concentration to counts
        # TODO -- these should be ints
        dO2_ext *= self.conc_conversion['oxygen']
        # dO2 = int(dO2)
        dLactate_ext = - lactate_transport*self.conc_conversion['lactate']

        # retrieve the results
        return {
            'internal_species': {
                'HIF': dHIF * interval,
                'lactate': dLactate * interval,
                'GFP': dGFP * interval,
            },
            'boundary': {
                'exchange': {
                    'oxygen': dO2_ext*interval,
                    'lactate': dLactate_ext*interval,
                }
            }
        }


def test_cell():
    total_time = 1000

    # create the process
    parameters = {
        'timestep': 0.1,
        'k_HIF_production_max': 0.9,  # default is 1. bifurcation ~ 0.7, 1.4
        'lactate_initial': 0.001,  # default 0.2
        'external_lactate_initial': 0.1,
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
    fig = plot_simulation_output(
        data,
        settings=settings,
        out_dir='out',
        filename='results'
    )
    fig.show()


if __name__ == '__main__':
    test_cell()
