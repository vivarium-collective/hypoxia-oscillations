import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_simulation_data(
        simulation_data,
        num_rows,
        skip_paths=[],
        show_nth_site=1,  # TODO -- make it so it shows every nth site
        out_dir='out',
        filename=None
):
    # Function to check if the current path should be skipped
    def should_skip(current_path):
        return any(all(k == current_path[i] for i, k in enumerate(path)) for path in skip_paths if len(path) <= len(current_path))

    # Function to recursively process the nested dictionary
    def process_node(node, path, ax):
        if should_skip(path):
            return  # Skip this path

        if isinstance(node, dict):
            for key, value in node.items():
                process_node(value, path + [key], ax)
        else:
            # Assuming 'node' is now a list or similar iterable of data points
            ax.plot(node, label=' - '.join(path))

    # Total number of cells
    num_cells = len(simulation_data)
    # Calculate number of columns based on number of rows and cells
    num_cols = -(-num_cells // num_rows)  # Ceiling division

    # Create a figure with a gridspec layout
    fig = plt.figure(figsize=(num_cols * 4.5, num_rows * 3))
    gs = GridSpec(num_rows, num_cols, figure=fig)

    # Dictionary to store all the plot handles for legend
    plot_handles = {}

    # Index for current cell
    cell_index = 0
    for cell_id, data in simulation_data.items():
        # Determine row and column index
        row_index = cell_index // num_cols
        col_index = cell_index % num_cols

        ax = fig.add_subplot(gs[row_index, col_index])
        ax.set_title(f'Cell ID: {cell_id}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_yscale('log')  # log scale

        process_node(data, [], ax)

        # Update handles for legend
        handles, labels = ax.get_legend_handles_labels()
        plot_handles.update(dict(zip(labels, handles)))

        cell_index += 1

    # Add a shared legend to the figure
    # Adjust the location and font size of the legend
    fig.legend(
        plot_handles.values(),
        plot_handles.keys(),
        loc='center right',
        # bbox_to_anchor=(1.00, 0.5),
        fontsize='large'
    )

    # Adjust layout to accommodate the external legend
    # rect : tuple (left, bottom, right, top)
    fig.tight_layout(rect=[0, 0, 0.8, 1])

    if filename:
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, filename)
        print(f"Writing {fig_path}")
        fig.savefig(fig_path, bbox_inches='tight')

    return fig

# Example usage
# simulation_data = {
#     'cell1': {'first': {'path': {'variable1': [0, 1, 2, 3, 4]}}, 'second
