import itertools
import numpy as np

from vivarium.core.process import Process, Step
from vivarium.core.composer import Composer
from vivarium.core.engine import Engine
from vivarium.plots.simulation_output import plot_simulation_output
from library.analyze import detect_oscillation_period
from plots.scan import save_heat_map
from scipy import constants

AVOGADRO = constants.N_A


class SimpleCell(Process):
    """
    Simple Cell
    ===========

    A simple cell model with HIF, lactate, and GFP, following this system of equations:
        dHIF = ksxs + ksx*(HIF^2)/(kpx + HIF^2) - kdx*HIF - kdsx*HIF*lactate
        dlactate = ksy*HIF^2/(kpy + HIF^2) - kdy*lactate
        dGFP = Vg*HIF^3/(kg+HIF^3) - dg*GFP

    Latex equations:
        \frac{dHIF}{dt} = k_{sxs} + \frac{k_{sx}HIF^2}{k_{px} + HIF^2} - k_{dx}HIF - k_{dsx}HIF*lactate
        \frac{dlactate}{dt} = \frac{k_{sy}HIF^2}{k_{py} + HIF^2} - k_{dy}lactate
        \frac{dGFP}{dt} = \frac{V_gHIF^3}{k_g+HIF^3} - d_gGFP

    """

    defaults = {
        'k_HIF_production_basal': 0.02,  # k_sxs
        'k_HIF_production_max': 0.9,  # k_sx
        'k_HIF_pos_feedback': 1,  # k_px
        'k_HIF_deg_basal': 0.2,  # k_dx
        'k_HIF_deg_lactate': 1,  # k_dsx
        'k_lactate_production': 0.01,  # 0.01 k_sy
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
        'k_MCT1': 1E-3,  # 1E-3 lactate import
        'k_MCT4': 1E-3,  # 1E-3 lactate export

        # oxygen consumption
        'external_oxygen_initial': 1.1,

        # oxygen exchange
        'o2_response_scaling': 1.0,
        'kmax_o2_deg': 1e-1,
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
            # TODO -- add parameter k_O2_consumption
            dO2_ext = - self.parameters['o2_response_scaling'] * (self.parameters['k_min_o2_deg'] + self.parameters['kmax_o2_deg'] / (hif_in + 1))
            # dO2_ext = - self.parameters['k_min_o2_deg'] - self.parameters['kmax_o2_deg'] / (
            #         (hif_in / self.parameters['HIF_threshold'])**self.parameters['hill_coeff_o2_deg'] + 1)

        # convert dO2 and lactate transport from concentration to counts
        # TODO -- these should be ints
        dO2_ext *= self.conc_conversion['oxygen']
        dLactate_ext = - lactate_transport * self.conc_conversion['lactate']

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


class SimpleEnvironment(Step):
    defaults = {
        'volume': 1E2,
    }
    def __init__(self, parameters=None):
        super().__init__(parameters)

    def ports_schema(self):
        return {
            'exchange': {
                'oxygen': {
                    '_default': 1.0,
                    '_emit': True,
                },
                'lactate': {
                    '_default': 0.1,
                    '_emit': True,
                }
            },
            'external': {
                'oxygen': {
                    '_default': 1.0,
                    '_emit': True,
                },
                'lactate': {
                    '_default': 0.1,
                    '_emit': True,
                }
            }
        }

    def next_update(self, _, states):
        exchange = states['exchange']
        delta_external = {}
        reset_exchange = {}
        for mol_id, counts in exchange.items():
            if counts != 0:
                delta_external[mol_id] = counts / self.parameters['volume']
                reset_exchange[mol_id] = {
                    '_value': 0,
                    '_updater': 'set'}

        return {
            'external': delta_external,
            'exchange': reset_exchange
        }


class CellEnvironment(Composer):
    defaults = {
        'cell': {},
        'environment': {},
    }

    def generate_processes(self, config):
        cell = SimpleCell(config['cell'])
        environment = SimpleEnvironment(config['environment'])
        return {
            'cell': cell,
            'environment': environment,
        }

    def generate_topology(self, config):
        return {
            'cell': {
                'boundary': ('boundary',),
                'internal_species': ('internal_species',),
            },
            'environment': {
                'exchange': ('boundary', 'exchange'),
                'external': ('boundary', 'external'),
            }
        }


def run_cell_env(total_time=1000, config=None):
    config = config or {}

    cell_env = CellEnvironment(config).generate()

    sim = Engine(composite=cell_env)
    sim.update(total_time)

    return sim.emitter.get_timeseries()


def test_cell_environment():
    data = run_cell_env(
        total_time=2500,
        config={
            'environment': {
                'volume': 2E2
            }
    })
    # print(data)

    # plot results
    settings = {}
    fig = plot_simulation_output(
        data,
        settings=settings,
        out_dir='out',
        filename='results')
    fig.show()


def test_lac_env():
    data = run_cell_env(
        total_time=2500,
        config={
            'environment': {
                'volume': 2E4
            }
        })
    # print(data)

    # plot results
    settings = {}
    fig = plot_simulation_output(
        data,
        settings=settings,
        out_dir='out',
        filename='lac_env_results')
    fig.show()


def run_cell(total_time=1000, parameters=None):
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
    return data


def test_cell():
    total_time = 1000

    # create the process
    parameters = {
        # 'timestep': 0.1,
        # 'k_HIF_production_max': 0.9,  # default is 1. bifurcation ~ 0.7, 1.4
        # 'lactate_initial': 0.001,  # default 0.2
        # 'external_lactate_initial': 0.1,
        # 'external_oxygen_initial': 1.1,
    }

    data = run_cell(total_time, parameters)

    # plot results
    settings = {}
    fig = plot_simulation_output(
        data,
        settings=settings,
        out_dir='out',
        filename='results')
    fig.show()


def scan_cell(total_time, parameter_scan):
    keys, values = zip(*parameter_scan.items())
    detailed_results = []  # List to store both results and parameter values

    for value_combination in itertools.product(*values):
        parameters = dict(zip(keys, value_combination))
        results = run_cell(total_time, parameters)
        oscillation_period = detect_oscillation_period(results['internal_species']['GFP'])

        # Store the detailed result with parameters and the outcome
        detailed_result = {
            'parameters': parameters,
            'results': results,
            'oscillation_period': oscillation_period
        }
        detailed_results.append(detailed_result)

    return detailed_results


def test_scan_cell_o2_lac():
    total_time = 1500
    parameter_scan = {
        'external_oxygen_initial': list(np.linspace(0,1.3,30)),
        'external_lactate_initial': list(np.linspace(0,2.6,15)),
    }
    detailed_results = scan_cell(total_time, parameter_scan)

    save_heat_map(
        detailed_results,
        parameter_scan,
        'external_oxygen_initial',
        'external_lactate_initial',
        out_dir='out',
        filename='heatmap.png')


def test_scan_cell_lac_hif():
    total_time = 1500
    parameter_scan = {
        'k_HIF_pos_feedback': list(np.linspace(0, 1.8, 20)),
        'external_lactate_initial': list(np.linspace(0, 1.5, 20)),
    }
    detailed_results = scan_cell(total_time, parameter_scan)

    save_heat_map(
        detailed_results,
        parameter_scan,
        'k_HIF_pos_feedback',
        'external_lactate_initial',
        out_dir='out',
        filename='scan_lac_HIF_feedback.png')

    for detailed_result in detailed_results:
        result = detailed_result['results']
        parameters = detailed_result['parameters']
        # plot results
        filename = f'{parameters}.png'
        settings = {}
        plot_simulation_output(
            result,
            settings=settings,
            out_dir='out/lac_hif_pos/',
            filename=filename)


def test_scan_cell_o2_scaling():
    total_time = 1500
    parameter_scan = {
        'o2_response_scaling': list(np.linspace(0,2.0,30)),
    }
    detailed_results = scan_cell(total_time, parameter_scan)

    for detailed_result in detailed_results:
        result = detailed_result['results']
        parameters = detailed_result['parameters']
        # plot results
        filename = f'o2results_{parameters}.png'
        settings = {}
        plot_simulation_output(
            result,
            settings=settings,
            out_dir='out/o2/',
            filename=filename)



if __name__ == '__main__':
    # test_cell()
    # test_scan_cell_o2_lac()
    # test_scan_cell_o2_scaling()
    # test_cell_environment()
    test_scan_cell_lac_hif()
    # test_lac_env()