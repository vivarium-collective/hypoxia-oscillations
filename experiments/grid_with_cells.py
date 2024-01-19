"""
Grid Experiment
"""
import random

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
    total_time=800,
    bin_size=1,
    bounds=None,
    depth=None,
    oxygen_clamp_value=None,
    density=0.9,  # 1.0 = 100% density
):
    n_snapshots = 6  # number of snapshots for temporal fields plot

    # cell parameters
    lactate_production = 1e-1

    # set defaults
    bounds = bounds or DEFAULT_BOUNDS
    bin_size = bin_size or DEFAULT_BIN_SIZE
    depth = depth or DEFAULT_DEPTH
    oxygen_clamp_value = oxygen_clamp_value or OXYGEN_CLAMP_VALUE

    kmax_o2_deg = 1e0

    # initialize composite dicts
    grid_processes = {'cells': {}}
    grid_topology = {'cells': {}}
    grid_initial_state = {'cells': {}}

    # initialize diffusion process
    config = {
        'bounds': bounds,
        'bin_size': bin_size,
        'depth': depth,
        'diffusion': {
            'lactate': 1E-2,  # cm^2 / day
            'oxygen': 1E-1,  # cm^2 / day
        },
        'clamp_edges': {'oxygen': oxygen_clamp_value}
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
    cell_config = {
        'k_lactate_production': lactate_production,
    }
    cell_composer = CompositeCell(cell_config)

    # add cells
    for x in range(bounds[0]):
        for y in range(bounds[1]):
            if random.random() <= density:

                # make the cell
                cell_id = f'[{x},{y}]'
                cell = cell_composer.generate({'cell_id': cell_id,
                                               'cell_config': {
                                                    'kmax_o2_deg': kmax_o2_deg,
                                               }})

                # add cell to grid
                grid_processes['cells'][cell_id] = cell['processes']
                grid_topology['cells'][cell_id] = cell['topology']
                grid_initial_state['cells'][cell_id] = {
                    'boundary': {
                        'location': [x*bin_size,y*bin_size]
                    }
                }

    # get initial state from diffusion process
    # field_state = diffusion_process.initial_state({'random': 1.0})
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
    data = sim.emitter.get_timeseries()

    # print(pf(data))

    # plot fields
    nth_timestep = int(total_time/(n_snapshots-1))
    temporal_fig = plot_fields_temporal(
        data['fields'],
        nth_timestep=nth_timestep,
        out_dir='out',
        filename=f'composite_fields_temporal'
    )
    temporal_fig.show()

    # plot results
    results_fig = plot_simulation_data(
        data['cells'],
        num_rows=bounds[0],
        num_cols=bounds[1],
        skip_paths=[
            # ['boundary'],
            # ['internal_store']
        ],
        filename=f'results_by_cell'
    )
    # results_fig.show()



if __name__ == '__main__':
    run_cell_grid()
