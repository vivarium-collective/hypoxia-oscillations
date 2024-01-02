"""
Grid Experiment
"""
from vivarium.core.engine import Engine, pf
from vivarium.plots.simulation_output import plot_simulation_output
from processes.diffusion_field import DiffusionField
from plots.field import plot_fields_temporal
from composites.composite_cell import CompositeCell


DEFAULT_BOUNDS = [4, 4]
DEFAULT_BIN_SIZE = 1
DEFAULT_DEPTH = 10


def run_cell_grid(
    total_time=60,
    bin_size=1,
    bounds=None,
    depth=None,
):
    # set defaults
    bounds = bounds or DEFAULT_BOUNDS
    bin_size = bin_size or DEFAULT_BIN_SIZE
    depth = depth or DEFAULT_DEPTH

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
    config = {}
    cell_composer = CompositeCell(config)

    # add cells
    for x in range(bounds[0]):
        for y in range(bounds[1]):
            # make the cell
            cell_id = f'[{x},{y}]'
            cell = cell_composer.generate({'cell_id': cell_id})

            # add cell to grid
            grid_processes['cells'][cell_id] = cell['processes']
            grid_topology['cells'][cell_id] = cell['topology']
            grid_initial_state['cells'][cell_id] = {
                'boundary': {
                    'location': [x*bin_size,y*bin_size]
                }
            }

    # get initial state from diffusion process
    field_state = diffusion_process.initial_state({'random': 1.0})
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
    temporal_fig = plot_fields_temporal(
        data['fields'],
        nth_timestep=10,
        out_dir='out',
        filename='composite_fields_temporal'
    )
    temporal_fig.show()

    # plot results
    settings = {
        'max_rows': bounds[0] * 5,
        'skip_ports': ['fields', 'dimensions']
    }
    results_fig = plot_simulation_output(
        data,
        settings=settings,
        out_dir='out',
        filename='cell_data'
    )
    results_fig.show()


if __name__ == '__main__':
    run_cell_grid()
