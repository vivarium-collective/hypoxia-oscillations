import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os


def save_heat_map(detailed_results, parameter_scan, param1, param2,
                  colormap='viridis', out_dir='out', filename='heatmap.png'):
    # Prepare parameter values for the axes
    param1_values = parameter_scan[param1]
    param2_values = parameter_scan[param2]

    # Initialize matrix for oscillation periods
    oscillation_matrix = np.zeros((len(param1_values), len(param2_values)))

    # Create mappings from parameter values to their indices for each parameter
    param1_to_index = {value: index for index, value in enumerate(param1_values)}
    param2_to_index = {value: index for index, value in enumerate(param2_values)}

    # Fill in the matrix with oscillation periods from detailed_results
    for result in detailed_results:
        param1_val = result['parameters'][param1]
        param2_val = result['parameters'][param2]
        osc_period = result['oscillation_period']
        i = param1_to_index[param1_val]
        j = param2_to_index[param2_val]
        oscillation_matrix[i, j] = osc_period

    # Generate the heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(oscillation_matrix.T, cmap=colormap, origin='lower')

    # Adjust the axes
    ax.set_xticks(np.arange(len(param1_values)))
    ax.set_yticks(np.arange(len(param2_values)))
    ax.set_xticklabels([str(round(val, 2)) for val in param1_values])
    ax.set_yticklabels([str(round(val, 2)) for val in param2_values])
    plt.xlabel(param1.replace('_', ' ').title())
    plt.ylabel(param2.replace('_', ' ').title())
    plt.title('Oscillation Period Heat Map')

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)  # Use 'im' here, the return value from imshow

    # Save the heatmap
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, filename), bbox_inches='tight')
    plt.close(fig)
