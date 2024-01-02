"""
Composite Cell
"""

from vivarium.core.composer import Composer
from processes.simple_cell import SimpleCell


class CompositeCell(Composer):
    defaults = {
        'cell_id': None,
        'cell_config': {},
    }

    def __init__(self, config):
        super().__init__(config)

    def generate_processes(self, config):
        cell_process = SimpleCell(config['cell_config'])
        return {
            'cell_process': cell_process
        }

    def generate_topology(self, config):
        return {
            'cell_process': {
                'internal_species': ('internal_store',),
                'boundary': ('boundary',),
            }
        }
