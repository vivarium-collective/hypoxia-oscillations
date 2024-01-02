from vivarium.core.engine import Engine, pf
from processes.simple_cell import SimpleCell
from processes.diffusion_field import DiffusionField
from plots.field import plot_fields_temporal
from vivarium.plots.simulation_output import plot_simulation_output

DEFAULT_BOUNDS = [4, 4]
DEFAULT_BIN_SIZE = 1


def run_cell_grid(
    total_time=60,
    bin_size=1,
    bounds=None,
):
    # set defaults
    bounds = bounds or DEFAULT_BOUNDS
    bin_size = bin_size or DEFAULT_BIN_SIZE

    # initialize composite dicts
    processes = {'cells': {}}
    topology = {'cells': {}}
    initial_state = {'cells': {}}

    # initialize diffusion process
    config = {
        'bounds': bounds,
        'bin_size': bin_size
    }
    diffusion_process = DiffusionField(config)

    # add diffusion process to composite
    processes['diffusion'] = diffusion_process
    topology['diffusion'] = {
        'fields': ('fields',),
        'dimensions': ('dimensions',),
        'cells': ('cells',),
    }

    # add cells
    for x in range(bounds[0]):
        for y in range(bounds[1]):
            # make the cell
            parameters = {}
            cell_process = SimpleCell(parameters)

            # add cell to composite
            cell_name = f'[{x},{y}]'
            processes['cells'][cell_name] = {'cell_process': cell_process}
            topology['cells'][cell_name] = {
                'cell_process': {
                    'internal_species': ('internal_store',),
                    'boundary': ('boundary',),
                }}
            initial_state['cells'][cell_name] = {'boundary': {'location': [x*bin_size,y*bin_size]}}

    # get initial state from diffusion process
    field_state = diffusion_process.initial_state({'random': 1.0})
    initial_state['fields'] = field_state['fields']

    # initialize simulation
    sim = Engine(
        initial_state=initial_state,
        processes=processes,
        topology=topology
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
    settings = {}
    results_fig = plot_simulation_output(
        data,
        settings=settings,
        out_dir='out',
        filename='cell_data'
    )
    results_fig.show()


if __name__ == '__main__':
    run_cell_grid()
