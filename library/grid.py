import numpy as np


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
