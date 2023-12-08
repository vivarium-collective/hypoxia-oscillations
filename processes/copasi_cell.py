from vivarium.core.process import Process
from vivarium.core.engine import Engine, pp
from basico import *


DEFAULT_MODEL_FILE = 'SPN_1_b.cps'


class COPASICell(Process):
    defaults = {
        'model_file': DEFAULT_MODEL_FILE,
        'boundary_molecules': ['EWG', 'HH', 'PH', 'PTC'],
        'time_step': 1.0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # Load the single cell model into Basico
        self.copasi_model_object = load_model(self.parameters['model_file'])
        all_species = get_species(model=self.copasi_model_object).index.tolist()
        self.external_species = [
            f'{mol_id}{side_id}_ext' for side_id in range(1, self.parameters['n_sides'] + 1)
            for mol_id in self.parameters['boundary_molecules']]
        self.internal_species = [
            mol_id for mol_id in all_species if mol_id not in self.external_species]


    def ports_schema(self):
        return {
            'external': {mol_id: {} for mol_id in self.external_species},
            'internal': {mol_id: {} for mol_id in self.internal_species},  # TODO output only
        }

    def next_update(self, endtime, states):

        internal = states['internal']
        external = states['external']

        # and set them in the model
        # for mol_id, value in internal.items():
        #     set_species(name=mol_id, initial_concentration=value, model=self.copasi_model_object)
        for mol_id, value in external.items():
            set_species(name=mol_id, initial_concentration=value, model=self.copasi_model_object)

        # run model for "endtime" length; we only want the state at the end of endtime, if we need more we can set intervals to a larger value
        timecourse = run_time_course(duration=endtime, intervals=1, update_model=True, model=self.copasi_model_object)

        # extract end values of concentrations from the model and set them in results (18 states)
        results = {}
        # for side_id in range(1, self.parameters['n_sides']+1):
        #     mol_ids = [f'{i}{side_id}' for i in self.parameters['boundary_molecules']]
        #     # get these values from the copasi model
        #     results[str(side_id)] = {
        #         mol_id: float(get_species(name=mol_id, exact=True, model=self.copasi_model_object).concentration[0])
        #         for mol_id in mol_ids}
        #
        # results['internal_species'] = {
        #     mol_id: float(get_species(name=mol_id, exact=True, model=self.copasi_model_object).concentration[0])
        #     for mol_id in self.internal_species}

        return results


def test_spn():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, f"../{DEFAULT_MODEL_FILE}")
    total_time = 1000
    config = {
        'model_file': model_path,
        'boundary_molecules': ['EWG', 'HH', 'PH', 'PTC'],
        'n_sides': 6,
        'time_step': 100
    }
    spn_process = SPNcell(config)

    ports = spn_process.ports_schema()
    print('PORTS')
    print(ports)

    sim = Engine(
        processes={'spn': spn_process},
        topology={'spn': {port_id: (port_id,) for port_id in ports.keys()}}
    )

    sim.update(total_time)

    data = sim.emitter.get_timeseries()
    print('RESULTS')
    print(pp(data))


if __name__ == '__main__':
    test_spn()
