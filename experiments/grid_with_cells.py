"""
Grid Experiment
"""
import random
import numpy as np

from vivarium.core.engine import Engine, pf
from processes.diffusion_field import DiffusionField
from plots.field import plot_fields_temporal
from plots.timeseries import plot_simulation_data
from composites.composite_cell import CompositeCell


DEFAULT_BOUNDS = [20, 20]
DEFAULT_BIN_SIZE = 1
DEFAULT_DEPTH = 10
OXYGEN_CLAMP_VALUE = 2  # 1.1


def run_cell_grid(
        total_time,
        bounds=DEFAULT_BOUNDS,
        bin_size=DEFAULT_BIN_SIZE,
        depth=DEFAULT_DEPTH,
        diffusion_constants=None,
        clamp_edges=False,
        density=1.0,
        perturb_cell_parameters=None,
        cell_parameters=None,
):
    # parameters
    perturb_cell_parameters = perturb_cell_parameters or {}
    cell_parameters = cell_parameters or {}

    # initialize composite dicts
    grid_processes = {'cells': {}}
    grid_topology = {'cells': {}}
    grid_initial_state = {'cells': {}}

    # initialize diffusion process
    config = {
        'bounds': bounds,
        'bin_size': bin_size,
        'depth': depth,
        'diffusion': diffusion_constants or {},
        'clamp_edges': clamp_edges
    }
    diffusion_process = DiffusionField(config)

    # add diffusion process to composite
    grid_processes['diffusion'] = diffusion_process
    grid_topology['diffusion'] = {
        'fields': ('fields',),
        'dimensions': ('dimensions',),
        'cells': ('cells',),
    }

    # make cell composer
    cell_config = {}
    cell_composer = CompositeCell(cell_config)

    # add cells
    for x in range(bounds[0]):
        for y in range(bounds[1]):
            if random.random() <= density:

                # get parameters
                parameters = cell_parameters.copy()

                # update parameters
                for param_id, spec in perturb_cell_parameters.items():
                    # sample from log normal distribution
                    scaling = np.random.lognormal(spec['loc'], spec['scale'])
                    if scaling < 0:
                        scaling = 0
                    parameters[param_id] = scaling
                    # parameters[param_id] = np.random.normal(spec['loc'], spec['scale'])

                # make the cell
                cell_id = f'[{x},{y}]'
                cell = cell_composer.generate({'cell_id': cell_id,
                                               'cell_config': parameters})

                # add cell to grid
                grid_processes['cells'][cell_id] = cell['processes']
                grid_topology['cells'][cell_id] = cell['topology']
                grid_initial_state['cells'][cell_id] = {
                    'boundary': {
                        'location': [x * bin_size, y * bin_size]
                    }
                }

    # get initial state from diffusion process
    # field_state = diffusion_process.initial_state({'random': 1.0})

    # TODO -- make configurable
    field_state = diffusion_process.initial_state({
        'random': {
            'lactate': 0.2,
            'oxygen': 2.2
        }})
    grid_initial_state['fields'] = field_state['fields']

    # initialize simulation
    sim = Engine(
        initial_state=grid_initial_state,
        processes=grid_processes,
        topology=grid_topology
    )

    # run simulation
    sim.update(total_time)

    # retrieve results
    return sim.emitter.get_timeseries()


def run_cells1():
    total_time = 800
    perturb_cell_parameters = {
        'o2_response_scaling': {'loc': 0.0, 'scale': 0.25},
    }

    data = run_cell_grid(
        total_time=total_time,
        perturb_cell_parameters=perturb_cell_parameters,
        cell_parameters={
            'kmax_o2_deg': 1e0,
            # 'k_lactate_production': 1e-1,
        },
        diffusion_constants={
            'lactate': 1E-2,  # cm^2 / day
            'oxygen': 1E-1,  # cm^2 / day
        },
        clamp_edges={
            'oxygen': OXYGEN_CLAMP_VALUE
        },
        density=0.9,
    )


    # plot fields
    n_snapshots = 6  # number of snapshots for temporal fields plot
    nth_timestep = int(total_time/(n_snapshots-1))
    temporal_fig = plot_fields_temporal(
        data['fields'],
        nth_timestep=nth_timestep,
        out_dir='out',
        filename=f'composite_fields_temporal1'
    )
    temporal_fig.show()

    # plot results
    results_fig = plot_simulation_data(
        data['cells'],
        num_rows=DEFAULT_BOUNDS[0],
        num_cols=DEFAULT_BOUNDS[1],
        filename=f'results_by_cell1'
    )
    # results_fig.show()


def run_cell2():
    pass


if __name__ == '__main__':
    run_cells1()
    # run_cell2()