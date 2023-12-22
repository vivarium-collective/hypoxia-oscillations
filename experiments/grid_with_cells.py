from vivarium.core.engine import Engine, pf
from processes.simple_cell import SimpleCell
from processes.diffusion import Diffusion
from plots.field import plot_fields_temporal
from vivarium.plots.simulation_output import plot_simulation_output


def test_cell_grid():
    total_time = 60
    bounds = [4, 4]
    bin_size = 1

    processes = {'cells': {}}
    topology = {'cells': {}}
    initial_state = {'cells': {}}

    # initialize diffusion field
    config = {
        'bounds': bounds,
        'bin_size': bin_size
    }
    diffusion_process = Diffusion(config)

    processes['diffusion'] = diffusion_process
    topology['diffusion'] = {
        'fields': ('fields',),
        'dimensions': ('dimensions',),
        'cells': ('cells',),
    }

    for x in range(bounds[0]):
        for y in range(bounds[1]):
            parameters = {}
            cell_process = SimpleCell(parameters)
            cell_name = f'[{x},{y}]'
            processes['cells'][cell_name] = {'cell_process': cell_process}
            topology['cells'][cell_name] = {
                'cell_process': {
                    'internal_species': ('internal_store',),
                    'boundary': ('boundary',),
                }}
            initial_state['cells'][cell_name] = {'boundary': {'location': [x*bin_size,y*bin_size]}}


    # initial state
    field_state = diffusion_process.initial_state({'random': 1.0})
    initial_state['fields'] = field_state['fields']

    # put it in a simulation
    sim = Engine(
        initial_state=initial_state,
        processes=processes,
        topology=topology
    )

    # run simulation
    sim.update(total_time)

    # get the results
    data = sim.emitter.get_timeseries()

    # print(pf(data))

    # plot results
    plot_fields_temporal(data['fields'], nth_timestep=10, out_dir='out', filename='composite_fields_temporal')


    # plot results
    settings = {}
    plot_simulation_output(
        data,
        settings=settings,
        out_dir='out',
        filename='cell_data'
    )


if __name__ == '__main__':
    test_cell_grid()
