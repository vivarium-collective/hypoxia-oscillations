import numpy as np
from scipy.fft import rfft, rfftfreq
# from scipy.signal import find_peaks
from scipy.signal import find_peaks, butter, filtfilt


def detect_oscillation_period(signal, sampling_rate=1, window_fraction=0.5, threshold_fraction=0.1):
    """
    Detects sustained oscillation in a time series signal using FFT and checks for persistence over time.

    :param signal: List or numpy array containing the signal values.
    :param sampling_rate: The sampling rate of the signal (default is 1).
    :param window_fraction: Fraction of the signal to be considered for checking sustained oscillation.
    :param threshold_fraction: Fraction of max magnitude to consider for peak detection.
    :return: Oscillation period if detected, otherwise None.
    """
    # Filter the signal to remove high-frequency noise
    b, a = butter(3, 0.05)
    filtered_signal = filtfilt(b, a, signal)

    # Perform the Fast Fourier Transform on the filtered signal
    signal_fft = rfft(filtered_signal)
    freqs = rfftfreq(len(filtered_signal), d=1. / sampling_rate)
    magnitude = np.abs(signal_fft)

    # Detect peaks in the magnitude spectrum
    peaks, _ = find_peaks(magnitude, height=np.max(magnitude) * threshold_fraction)

    if peaks.size > 0:
        dominant_peak = peaks[np.argmax(magnitude[peaks])]
        dominant_freq = freqs[dominant_peak]
        oscillation_period = 1 / dominant_freq if dominant_freq > 0 else None

        # Windowed analysis to check for sustained oscillations
        window_size = int(len(signal) * window_fraction)
        start_index = 0
        end_index = window_size

        while end_index <= len(signal):
            windowed_signal = signal[start_index:end_index]
            windowed_fft = rfft(windowed_signal)
            windowed_magnitude = np.abs(windowed_fft)
            windowed_peaks, _ = find_peaks(windowed_magnitude, height=np.max(windowed_magnitude) * threshold_fraction)

            if not windowed_peaks.size > 0:
                # If no significant peaks in the window, consider it not sustained
                return None

            start_index += window_size
            end_index += window_size

        return oscillation_period
    else:
        return None