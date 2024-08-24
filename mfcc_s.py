import scipy
from scipy.fftpack import dct
import numpy as np
import scipy.io.wavfile as wav
import librosa
from matplotlib import pyplot as plt
from IPython.display import Audio

# Define a function to convert from Hertz to Mel scale
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

# Define a function to convert from Mel scale to Hertz
def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)
def delta(cepstral_coefficients, tau=1):
    # Pad the array at the beginning and end to handle the boundary conditions
    padded = np.pad(cepstral_coefficients, ((tau, tau), (0, 0)), mode='edge')
    # Compute delta features
    delta_features = padded[tau + 1: ] - padded[: -tau - 1]
    return delta_features
# Read the uploaded audio file
# file_path = 'trimmed_test1.wav'
#file_path = 'zero.wav'
def get_MFCC(fp):
    file_path=fp
    sampling_rate, signal = wav.read(file_path)

    # Pre-emphasis parameter
    pre_emphasis = 0.97

    # Apply pre-emphasis filter
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # Parameters for framing
    frame_size = 0.025  # Frame size of 25 ms
    frame_stride = 0.01  # Frame stride of 10 ms

    # Convert frame size and stride from seconds to samples
    frame_length, frame_step = frame_size * sampling_rate, frame_stride * sampling_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Ensure at least one frame

    # Padding for incomplete last frame
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    # Frame the signal
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Windowing
    frames *= scipy.signal.windows.hann(frame_length)

    # Compute FFT
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude spectrum
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power spectrum

    # Compute Mel filter banks
    nfilt = 40  # Number of filters
    low_freq_hz = 133.33
    high_freq_hz = 6855.4976
    low_freq_mel = hz_to_mel(low_freq_hz)
    high_freq_mel = hz_to_mel(high_freq_hz)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced points in Mel scale
    hz_points = mel_to_hz(mel_points)
    bin = np.floor((NFFT + 1) * hz_points / sampling_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # Left
        f_m = int(bin[m])             # Center
        f_m_plus = int(bin[m + 1])    # Right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical stability
    filter_banks = 20 * np.log10(filter_banks)  # Convert to dB

    # DCT to obtain MFCC
    num_ceps = 13
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]  # Keep dimensions 2-13

    # Define a function to calculate deltas


    tau = 1
    deltas = delta(mfcc, tau)
    delta_delta = delta(deltas, tau)
    final_mfcc=np.concatenate((mfcc, deltas, delta_delta), axis=1)
    y, sr = librosa.load(fp, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = mfccs.T
    deltas2 = delta(mfccs, tau)
    delta_delta2 = delta(deltas2, tau)
    final_mfccs = np.concatenate((mfccs, deltas2, delta_delta2), axis=1)
    return final_mfccs#final_mfcc
    # Play the audio