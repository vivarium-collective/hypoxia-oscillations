import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks


def detect_oscillation_period(signal, sampling_rate=1):
    """
    Detects oscillation in a time series signal using FFT.

    :param signal: List or numpy array containing the signal values.
    :param sampling_rate: The sampling rate of the signal (default is 1).
    :return: Oscillation period if detected, otherwise None.
    """
    # Perform the Fast Fourier Transform on the signal
    signal_fft = rfft(signal)
    # Get the corresponding frequencies
    freqs = rfftfreq(len(signal), d=1. / sampling_rate)
    # Calculate the magnitude of the FFT
    magnitude = np.abs(signal_fft)

    # Find peaks in the magnitude of the FFT which correspond to dominant frequencies
    peaks, _ = find_peaks(magnitude, height=np.max(magnitude) * 0.1)

    if peaks.size > 0:
        # Find the peak with the maximum magnitude (dominant frequency)
        dominant_peak = peaks[np.argmax(magnitude[peaks])]
        dominant_freq = freqs[dominant_peak]

        # Calculate the period of oscillation as the reciprocal of the dominant frequency
        oscillation_period = 1 / dominant_freq if dominant_freq > 0 else None
        return oscillation_period
    else:
        # No significant peaks detected
        return None


# # Example usage with a sample signal
# sampling_rate = 1  # Assuming one sample per unit time, adjust as per your data
# signal = [...]  # Replace [...] with your actual signal data, e.g., internal_species['HIF']
# oscillation_period = detect_oscillation_period(signal, sampling_rate)
#
# print(f"Oscillation Period: {oscillation_period if oscillation_period else 'None'}")
