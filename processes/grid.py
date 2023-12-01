from vivarium.core.process import Process, Step


class Grid(Process):
    defaults = {
        'bounds': [10, 10],  # TODO -- what units?
        'bin_size': 1,
        'field_molecules': [
            'oxygen',
            'lactate',
            'glutamine',
        ]
    }

    def __init__(self, parameters):
        super().__init__(parameters)
        self.bounds = self.parameters['bounds']
        self.field_molecules = self.parameters['field_molecules']
        bin_size = self.parameters['bin_size']

        n_bins_x = self.bounds[0]/bin_size  # TODO -- assert this divides cleanly
        n_bins_y = self.bounds[1] / bin_size

    def ports_schema(self):
        return {
            'cells': {},
            'fields': {
                mol_id: {} for mol_id in self.field_molecules

            }
        }

    def next_update(self, timestep, states):
        # set the states

        # run the simulation

        # retrieve the results

        return {}


