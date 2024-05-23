"""
scratch code for debugging the simple cell model
"""

import os
import copy
import numpy as np
from scipy.integrate import odeint
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, find_peaks, get_window
import matplotlib.pyplot as plt

import warnings
warnings.warn("This module is for debugging purposes only and should not be imported in production code.", ImportWarning)


DEFAULT_PARAMS = {
    'oxygen_ex': 1.1,
    'lactate_ex': 0.1,
    'k_HIF_production_basal': 0.02,
    'k_HIF_production_max': 0.9,
    'k_HIF_pos_feedback': 1,
    'k_HIF_deg_basal': 0.2,
    'k_HIF_deg_lactate': 1,
    'k_lactate_production': 0.01,
    'k_lactate_production_reg': 1,
    'k_lactate_deg_basal': 0.01,
    'k_GFP_production_constantFP_production': 1,
    'k_GFP_production_constant': 0.05,
    'k_GFP_deg': 0.1,
    'k_MCT1': 1E-3,
    'k_MCT4': 1E-3,
    'o2_response_scaling': 1.0,
    'kmax_o2_deg': 1e-1,
    'k_min_o2_deg': 1e-2,
    'conc_conversion_oxygen': 1.0,
    'conc_conversion_lactate': 1.0
}


def system_of_odes(
        y, t,
        oxygen_ex=1.1,
        lactate_ex=0.1,
        k_HIF_production_basal=0.02,
        k_HIF_production_max=0.9,
        k_HIF_pos_feedback=1,
        k_HIF_deg_basal=0.2,
        k_HIF_deg_lactate=1,
        k_lactate_production=0.01,
        k_lactate_production_reg=1,
        k_lactate_deg_basal=0.01,
        k_GFP_production_constantFP_production=1,
        k_GFP_production_constant=0.05,
        k_GFP_deg=0.1,
        k_MCT1=1E-3,
        k_MCT4=1E-3,
        o2_response_scaling=1.0,
        kmax_o2_deg=1e-1,
        k_min_o2_deg=1e-2,
        conc_conversion_oxygen=1.0,
        conc_conversion_lactate=1.0
):
    # Unpack variables
    hif_in, lactate_in, gfp_in = y

    # System equations
    hif_production = (k_HIF_production_basal +
                      k_HIF_production_max * hif_in ** 2 /
                      (k_HIF_pos_feedback + hif_in ** 2))
    hif_degradation = (k_HIF_deg_basal * hif_in * oxygen_ex +
                       k_HIF_deg_lactate * hif_in * lactate_in)
    dHIF = hif_production - hif_degradation

    lactate_production = (k_lactate_production * hif_in ** 2 /
                          (k_lactate_production_reg + hif_in ** 2))
    lactate_degradation = k_lactate_deg_basal * lactate_in
    lactate_transport = k_MCT1 * lactate_ex - k_MCT4 * lactate_in
    dLactate = lactate_production - lactate_degradation + lactate_transport

    # lactate_total_produced = (lactate_production - lactate_degradation) - lac_0

    gfp_production = (k_GFP_production_constantFP_production * hif_in ** 3 /
                      (k_GFP_production_constant + hif_in ** 3))
    gfp_degradation = k_GFP_deg * gfp_in
    dGFP = gfp_production - gfp_degradation

    dO2_ext = 0
    if oxygen_ex > 0:
        dO2_ext = -o2_response_scaling * (k_min_o2_deg + kmax_o2_deg / (hif_in + 1))

    # Convert from concentration to counts (or vice versa)
    dO2_ext *= conc_conversion_oxygen
    dLactate_ext = -lactate_transport * conc_conversion_lactate

    return [
        dHIF,
        dLactate,
        dGFP,
    ]


def plot_ode_solution(t, solution, lactate_production_rates=None, lactate_degradation_rates=None):
    # Unpack the solution
    hif_in, lactate_in, gfp_in = solution.T  # Transpose to unpack by columns

    # Create the plots
    n_plots = 3
    if lactate_production_rates is not None:
        n_plots +=1
    if lactate_degradation_rates is not None:
        n_plots +=1
    fig, axs = plt.subplots(n_plots, 1, figsize=(6, 4), sharex=True)  # Set up subplots with shared x-axis

    # Plot HIF
    axs[0].plot(t, hif_in, label='HIF')
    axs[0].set_ylabel('HIF\nconc')
    # axs[0].legend()

    # Remove x-ticks for the first two plots
    axs[0].tick_params(labelbottom=False)  # Hide x-ticks for the first plot

    # Plot internal Lactate
    axs[1].plot(t, lactate_in, label='Lactate (internal)')
    axs[1].set_ylabel('Lac\nconc')
    # axs[1].legend()
    axs[1].tick_params(labelbottom=False)  # Hide x-ticks for the second plot

    # Plot GFP
    axs[2].plot(t, gfp_in, label='GFP')
    # axs[2].set_xlabel('Time')  # Only this plot will show the x-axis label
    axs[2].set_ylabel('GFP\nconc')
    # axs[2].legend()

    # Plot lactate production and degradation rates
    if lactate_production_rates is not None:
        axs[3].plot(t, lactate_production_rates, label='Lactate Production Rate')
        axs[3].set_ylabel('Lac\nProd\nRate')
        # axs[3].legend()
    if lactate_degradation_rates is not None:
        axs[4].plot(t, lactate_degradation_rates, label='Lactate Degradation Rate')
        axs[4].set_ylabel('Lac\nDeg\nRate')
        # axs[4].legend()

    # time on the bottom axis
    axs[n_plots-1].set_xlabel('Time')  # Only this plot will show the x-axis label

    plt.tight_layout()  # Adjust the layout to make room for all the plots
    plt.show()


def detect_oscillation_period(
        signal,
        timesteps,
        window_fraction=0.5,
        threshold_fraction=0.1,
        overlap_fraction=0.1
):
    # Calculate actual sampling intervals
    sampling_intervals = np.diff(timesteps)
    average_sampling_interval = np.mean(sampling_intervals)

    # Constants and window setup
    window_size = int(len(signal) * window_fraction)
    step_size = int(window_size * (1 - overlap_fraction))
    window_function = get_window('hann', window_size)
    periods = []

    # Iterate over each window segment of the signal
    for start_index in range(0, len(signal) - window_size + 1, step_size):
        # Apply window function to the segment
        windowed_signal = signal[start_index:start_index + window_size] * window_function
        # Perform FFT
        windowed_fft = rfft(windowed_signal)
        freqs = rfftfreq(window_size, d=average_sampling_interval)
        magnitude = np.abs(windowed_fft)

        # Detect peaks and find the dominant frequency
        peak_threshold = np.max(magnitude) * threshold_fraction
        peaks, _ = find_peaks(magnitude, height=peak_threshold)
        if peaks.size > 0:
            dominant_peak = peaks[np.argmax(magnitude[peaks])]
            dominant_freq = freqs[dominant_peak]
            if dominant_freq > 0:
                period = 1 / dominant_freq
                periods.append(period)

    # Calculate the average period if any periods were detected
    if periods:
        return np.mean(periods)
    else:
        return None


def plot_phase_diagram(
        param1_name,
        param1_range,
        param2_name,
        param2_range,
        total_time=1000,
        resolution=50,
        filename='phase_diagram.png',
):
    param1_values = np.linspace(*param1_range, resolution)
    param2_values = np.linspace(*param2_range, resolution)
    period_matrix = np.zeros((resolution, resolution))
    params = copy.deepcopy(DEFAULT_PARAMS)

    for i, param1 in enumerate(param1_values):
        for j, param2 in enumerate(param2_values):
            # Setup initial conditions and time array
            y0 = [0.1, 0.001, 0.0]
            t = np.linspace(0, total_time, int(total_time))

            # Update the two parameters we're varying
            params[param1_name] = param1
            params[param2_name] = param2

            # Solve ODE
            solution = odeint(system_of_odes, y0, t, args=tuple(params.values()))

            # Detect oscillation period
            signal = solution[:, 0]  # Assuming we're interested in the HIF component for oscillation detection
            period = detect_oscillation_period(signal, t)

            # Store the period in the matrix
            period_matrix[i, j] = period if period is not None else 0


            # # print(f'{param1_name}={param1}, {param2_name}={param2}, Period={period}')
            # if param2 < 0.27 and param2 > 0.23 and param1 < 0.55 and param1 > 0.45:
            #     print(f'{param1_name}={param1} i={i}, {param2_name}={param2} j={j}, Period={period}')
            #     plot_ode_solution(t, solution)

    # Plotting the phase diagram
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(period_matrix, origin='lower', extent=[*param2_range, *param1_range], aspect='auto')
    fig.colorbar(cax, label='Oscillation Period')
    ax.set_xlabel(param2_name)
    ax.set_ylabel(param1_name)
    ax.set_title('Phase Diagram of Oscillation Periods')

    fig_path = os.path.join('out', filename)
    fig.savefig(fig_path)
    plt.close(fig)

    return period_matrix


def run_zoom1():
    # Define the parameter ranges
    k_HIF_deg_lactate_range = (0.0, 3.0)
    lactate_ex_range = (0.0, 1.0)

    # Define the total time for simulation and resolution of the scan
    total_time = 5000  # Total time to simulate the ODEs
    resolution = 50  # Number of points to evaluate in each parameter range

    # Call the function to plot the phase diagram
    period_matrix = plot_phase_diagram(
        param1_name='k_HIF_deg_lactate',
        param1_range=k_HIF_deg_lactate_range,
        param2_name='lactate_ex',
        param2_range=lactate_ex_range,
        total_time=total_time,
        resolution=resolution,
        filename='zoom1.png'
    )

    x=1


def run_zoom2():
    # Define the parameter ranges
    k_HIF_deg_lactate_range = (0.0, 2.5)
    lactate_ex_range = (0.1, 0.3)

    # Define the total time for simulation and resolution of the scan
    total_time = 5000  # Total time to simulate the ODEs
    resolution = 50  # Number of points to evaluate in each parameter range

    # Call the function to plot the phase diagram
    plot_phase_diagram(
        param1_name='k_HIF_deg_lactate',
        param1_range=k_HIF_deg_lactate_range,
        param2_name='lactate_ex',
        param2_range=lactate_ex_range,
        total_time=total_time,
        resolution=resolution,
        filename='zoom2.png'
    )

def run_zoom3():
    # Define the parameter ranges
    k_HIF_deg_lactate_range = (0.4, 0.6)
    lactate_ex_range = (0.15, 0.35)

    # Define the total time for simulation and resolution of the scan
    total_time = 5000  # Total time to simulate the ODEs
    resolution = 5  # Number of points to evaluate in each parameter range

    # Call the function to plot the phase diagram
    plot_phase_diagram(
        param1_name='k_HIF_deg_lactate',
        param1_range=k_HIF_deg_lactate_range,
        param2_name='lactate_ex',
        param2_range=lactate_ex_range,
        total_time=total_time,
        resolution=resolution,
        filename='zoom3.png'
    )


def run_single(params_set):
    total_time = 5000

    params = copy.deepcopy(DEFAULT_PARAMS)

    # Setup initial conditions and time array
    y0 = [0.1, 0.001, 0.0]
    t = np.linspace(0, total_time, int(total_time))

    # Update the two parameters we're varying
    params.update(params_set)

    # Solve ODE
    solution = odeint(system_of_odes, y0, t, args=tuple(params.values()))
    plot_ode_solution(t, solution)


if __name__ == '__main__':
    run_zoom1()
    run_zoom2()
    # run_zoom3()

    #
    # run_single({
    #     'k_HIF_deg_lactate': 0.555,
    #     'lactate_ex': 0.25
    # })

    # run_single({
    #     'k_HIF_deg_lactate': 0.555,
    #     'lactate_ex': 0.25
    # })
    # run_single({
    #     'k_HIF_deg_lactate': 0.5,
    #     'lactate_ex': 0.25
    # })

    # run_single({
    #     'k_HIF_deg_lactate': 0.5612244897959184,
    #     'lactate_ex': 0.25918367346938775}
    # )
