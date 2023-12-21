"""
===============
Diffusion Field
===============

Diffuses and decays molecular concentrations in a 2D field.
"""

import os
import copy
import cv2
import numpy as np
from scipy import constants

from vivarium.core.serialize import Quantity
from vivarium.core.process import Process
from vivarium.core.engine import Engine, pf
from vivarium.library.units import units, remove_units


# laplacian kernel for diffusion
LAPLACIAN_2D = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
AVOGADRO = constants.N_A
CONCENTRATION_UNIT = units.ng / units.mL
LENGTH_UNIT = units.um


def get_bin_site(location, n_bins, bounds):
    bin_site_no_rounding = np.array([
        location[0] * n_bins[0] / bounds[0],
        location[1] * n_bins[1] / bounds[1]
    ])
    bin_site = tuple(
        np.floor(bin_site_no_rounding).astype(int) % n_bins)
    return bin_site


def get_bin_volume(n_bins, bounds, depth):
    total_volume = (depth * bounds[0] * bounds[1]) * 1e-15  # (L)
    return total_volume / (n_bins[0] * n_bins[1])


class Diffusion(Process):

    defaults = {
        # parameters for the lattice dimensions
        'bounds': [10, 10],
        'bin_size': 1,
        'depth': 1.0,

        # molecules
        'molecules': ['lactate', 'oxygen'],

        # diffusion
        'default_diffusion_dt': 0.1,
        'default_diffusion_rate': 1e-1,

        # specific diffusion rates
        'diffusion': {
            'lactate': 1E-3,  # units.cm * units.cm / units.day,
            'oxygen': 1E-1,  # units.cm * units.cm / units.day,
        },
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # parameters
        self.molecule_ids = self.parameters['molecules']
        self.bounds = [b for b in self.parameters['bounds']]
        self.n_bins = [
            int(self.bounds[0] / self.parameters['bin_size']),
            int(self.bounds[1] / self.parameters['bin_size'])
        ]
        self.depth = self.parameters['depth']
        if isinstance(self.depth, Quantity):
            self.depth = self.depth.to(LENGTH_UNIT).magnitude

        # get diffusion rates
        diffusion_rate = self.parameters['default_diffusion_rate']
        bins_x = self.n_bins[0]
        bins_y = self.n_bins[1]
        length_x = self.bounds[0]
        length_y = self.bounds[1]
        dx = length_x / bins_x
        dy = length_y / bins_y
        dx2 = dx * dy

        # general diffusion rate
        self.diffusion_rate = diffusion_rate / dx2

        # diffusion rates for each individual molecules
        self.molecule_specific_diffusion = {
            mol_id: diff_rate/dx2
            for mol_id, diff_rate in self.parameters['diffusion'].items()}

        # get diffusion timestep
        diffusion_dt = 0.5 * dx ** 2 * dy ** 2 / (2 * diffusion_rate * (dx ** 2 + dy ** 2))
        self.diffusion_dt = min(diffusion_dt, self.parameters['default_diffusion_dt'])

        # get bin volume, to convert between counts and concentration
        self.bin_volume = get_bin_volume([bins_x, bins_y], self.bounds, self.depth)

    def initial_state(self, config=None):
        if config is None:
            config = {}
        if 'random' in config:
            max = config.get('random', 1)
            fields = {
                field: max * self.random_field()
                for field in self.parameters['molecules']}
        elif 'uniform' in config:
            fields = {
                field: config['uniform'] * self.ones_field()
                for field in self.parameters['molecules']}
        else:
            fields = {
                field: self.ones_field()
                for field in self.parameters['molecules']}
        return {
            'fields': fields,
            'cells': {},
        }

    def ports_schema(self):
        schema = {}

        # cells
        local_concentration_schema = {
            molecule: {
                '_default': 0.0}
            for molecule in self.parameters['molecules']}
        schema['cells'] = {
            '*': {
                'boundary': {
                    'location': {
                        '_default': [
                            0.5 * bound for bound in self.parameters['bounds']],
                     },
                    'external': local_concentration_schema
                }}}

        # fields
        fields_schema = {
            'fields': {
                field: {
                    '_default': self.ones_field(),
                    '_updater': 'nonnegative_accumulate',
                    '_emit': True,
                }
                for field in self.parameters['molecules']
            },
        }
        schema.update(fields_schema)

        # dimensions
        dimensions_schema = {
            'dimensions': {
                'bounds': {
                    '_value': self.bounds,
                    '_updater': 'set',
                    '_emit': True,
                },
                'bin_size': {
                    '_value': self.parameters['bin_size'],
                    '_updater': 'set',
                    '_emit': True,
                },
                'depth': {
                    '_value': self.depth,
                    '_updater': 'set',
                    '_emit': True,
                }
            },
        }
        schema.update(dimensions_schema)
        return schema

    def next_update(self, timestep, states):
        fields = states['fields']
        cells = states['cells']

        # degrade and diffuse
        fields_new = copy.deepcopy(fields)
        # fields_new = self.degrade_fields(fields_new, timestep)
        fields_new = self.diffuse_fields(fields_new, timestep)

        # get delta_fields
        delta_fields = {
            mol_id: fields_new[mol_id] - field
            for mol_id, field in fields.items()}

        # get each agent's local environment
        local_environments = self.set_local_environments(cells, fields_new)

        update = {'fields': delta_fields}
        if local_environments:
            update.update({'cells': local_environments})

        return update

    def get_bin_site(self, location):
        return get_bin_site(
            [l.to(LENGTH_UNIT).magnitude for l in location],
            self.n_bins,
            self.bounds)

    def get_single_local_environments(self, specs, fields):
        bin_site = self.get_bin_site(specs['location'])
        local_environment = {}
        for mol_id, field in fields.items():
            local_environment[mol_id] = field[bin_site]
        return local_environment

    def set_local_environments(self, cells, fields):
        local_environments = {}
        if cells:
            for agent_id, specs in cells.items():
                local_environments[agent_id] = {'boundary': {'external': {}}}
                cell_environment = self.get_single_local_environments(specs['boundary'], fields)
                local_environments[agent_id]['boundary']['external'] = {
                    mol_id: {
                        '_value': value,
                        '_updater': 'set'  # this overrides the default updater
                    } for mol_id, value in cell_environment.items()
                }
        return local_environments

    def ones_field(self):
        return np.ones((self.n_bins[0], self.n_bins[1]),dtype=np.float64)

    def random_field(self):
        return np.random.rand(
            self.parameters['n_bins'][0],
            self.parameters['n_bins'][1])

    def diffuse(self, field, timestep, diffusion_rate):
        """ diffuse a single field """
        t = 0.0
        dt = min(timestep, self.diffusion_dt)
        diffusion_rate_dt = diffusion_rate * dt
        while t < timestep:
            result = cv2.filter2D(field, -1, LAPLACIAN_2D)
            # result = convolve(field, LAPLACIAN_2D, mode='reflect')
            field += diffusion_rate_dt * result
            t += dt
        return field

    def diffuse_fields(self, fields, timestep):
        """ diffuse fields in a fields dictionary """
        for mol_id, field in fields.items():
            diffusion_rate = self.molecule_specific_diffusion.get(mol_id, self.diffusion_rate)
            # run diffusion if molecule field is not uniform
            if len(set(field.flatten())) != 1:
                fields[mol_id] = self.diffuse(field, timestep, diffusion_rate)
        return fields


def test_fields():
    total_time = 30

    # initialize process
    fields = Diffusion()

    # make the process
    field = Diffusion()

    # put it in a simulation
    sim = Engine(
        initial_state=field.initial_state({}),
        processes={'diffusion_process': field},
        topology={'diffusion_process': {
            'fields': ('fields',),
            'cells': ('cells',),
            'dimensions': ('dimensions',),
        }}
    )

    # run simulation
    sim.update(total_time)

    # get the results
    data = sim.emitter.get_timeseries()

    # print results
    print(pf(data))


if __name__ == '__main__':
    test_fields()
