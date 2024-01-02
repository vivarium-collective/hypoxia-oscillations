"""
Composite Cell
"""

from vivarium.core.composer import Composer
from processes.simple_cell import SimpleCell
from processes.local_field import LocalField


class CompositeCell(Composer):
    defaults = {
        'cell_id': None,
        'cell_config': {},
        'boundary_path': ('boundary',),
        'field_path': ('..', '..', 'fields',),
        'dimensions_path': ('..', '..', 'dimensions',),
    }

    def __init__(self, config):
        super().__init__(config)

    def generate_processes(self, config):
        cell_process = SimpleCell(config['cell_config'])
        local_field = LocalField()
        return {
            'cell_process': cell_process,
            'local_field': local_field,
        }

    def generate_topology(self, config):
        boundary_path = config['boundary_path']
        field_path = config['field_path']
        dimensions_path = config['dimensions_path']

        return {
            'cell_process': {
                'internal_species': ('internal_store',),
                'boundary': ('boundary',),
            },
            'local_field': {
                'exchanges': boundary_path + ('exchange',),
                'location': boundary_path + ('location',),
                'fields': field_path,
                'dimensions': dimensions_path,
            },
        }
