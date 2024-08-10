import os
import sys
import numpy as np
import scipy.signal as scsg
import matplotlib.pyplot as plt
import math

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import stft
import utilFunctions as UF

eps = np.finfo(float).eps

"""
A4-Part-4: Computing onset detection function (Optional)

Write a function to compute a simple onset detection function (ODF) using the STFT. Compute two ODFs 
one for each of the frequency bands, low and high. The low frequency band is the set of all the 
frequencies between 0 and 3000 Hz and the high frequency band is the set of all the frequencies 
between 3000 and 10000 Hz (excluding the boundary frequencies in both the cases). 

A brief description of the onset detection function can be found in the pdf document (A4-STFT.pdf, 
in Relevant Concepts section) in the assignment directory (A4). Start with an initial condition of 
ODF(0) = 0 in order to make the length of the ODF same as that of the energy envelope. Remember to 
apply a half wave rectification on the ODF. 

The input arguments to the function are the wav file name including the path (inputFile), window 
type (window), window length (M), FFT size (N), and hop size (H). The function should return a numpy 
array with two columns, where the first column is the ODF computed on the low frequency band and the 
second column is the ODF computed on the high frequency band.

Use stft.stftAnal() to obtain the STFT magnitude spectrum for all the audio frames. Then compute two 
energy values for each frequency band specified. While calculating frequency bins for each frequency 
band, consider only the bins that are within the specified frequency range. For example, for the low 
frequency band consider only the bins with frequency > 0 Hz and < 3000 Hz (you can use np.where() to 
find those bin indexes). This way we also remove the DC offset in the signal in energy envelope 
computation. The frequency corresponding to the bin index k can be computed as k*fs/N, where fs is 
the sampling rate of the signal.

To get a better understanding of the energy envelope and its characteristics you can plot the envelopes 
together with the spectrogram of the signal. You can use matplotlib plotting library for this purpose. 
To visualize the spectrogram of a signal, a good option is to use colormesh. You can reuse the code in
sms-tools/lectures/4-STFT/plots-code/spectrogram.py. Either overlay the envelopes on the spectrogram 
or plot them in a different subplot. Make sure you use the same range of the x-axis for both the 
spectrogram and the energy envelopes.

NOTE: Running these test cases might take a few seconds depending on your hardware.

Test case 1: Use piano.wav file with window = 'blackman', M = 513, N = 1024 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 69 (69 samples) and of the high frequency 
band span from 70 to 232 (163 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 2: Use piano.wav file with window = 'blackman', M = 2047, N = 4096 and H = 128 as input. 
The bin indexes of the low frequency band span from 1 to 278 (278 samples) and of the high frequency 
band span from 279 to 928 (650 samples). To numerically compare your output, use loadTestCases.py
script to obtain the expected output.

Test case 3: Use sax-phrase-short.wav file with window = 'hamming', M = 513, N = 2048 and H = 256 as 
input. The bin indexes of the low frequency band span from 1 to 139 (139 samples) and of the high 
frequency band span from 140 to 464 (325 samples). To numerically compare your output, use 
loadTestCases.py script to obtain the expected output.

In addition to comparing results with the expected output, you can also plot your output for these 
test cases. For test case 1, you can clearly see that the ODFs have sharp peaks at the onset of the 
piano notes (See figure in the accompanying pdf). You will notice exactly 6 peaks that are above 
10 dB value in the ODF computed on the high frequency band. 
"""

def computeODF(inputFile, window, M, N, H):
    """
    Inputs:
            inputFile (string): input sound file (monophonic with sampling rate of 44100)
            window (string): analysis window type (choice of rectangular, triangular, hanning, hamming,
                blackman, blackmanharris)
            M (integer): analysis window size (odd integer value)
            N (integer): fft size (power of two, bigger or equal than than M)
            H (integer): hop size for the STFT computation
    Output:
            The function should return a numpy array with two columns, where the first column is the ODF
            computed on the low frequency band and the second column is the ODF computed on the high
            frequency band.
            ODF[:,0]: ODF computed in band 0 < f < 3000 Hz
            ODF[:,1]: ODF computed in band 3000 < f < 10000 Hz
    """

    fs, x = UF.wavread(inputFile)
    w = scsg.windows.get_window(window, M, fftbins=M % 2 == 0)
    mX, pX = stft.stftAnal(x, w, N, H)

    # convert magnitude to linear scale
    mX = 10 ** (mX / 20.0)

    # frequency axis
    freqs = np.linspace(0, fs / 2, (N // 2) + 1, endpoint=True)

    # frequency bin indices for the specified bands
    k1 = [int(N * f / fs) for f in freqs if 0 < f < 3000]
    k2 = [int(N * f / fs) for f in freqs if 3000 < f < 10000]

    # energy envelopes in dB
    energy_envelope_low_db = 10 * np.log10(np.sum(mX[:, k1] ** 2, axis=1) + eps)
    energy_envelope_high_db = 10 * np.log10(np.sum(mX[:, k2] ** 2, axis=1) + eps)

    # onset detection functions
    odf_low_db = np.diff(energy_envelope_low_db, prepend=energy_envelope_low_db[0])
    odf_low_db = np.maximum(odf_low_db, 0)  # half-wave rectifier
    odf_high_db = np.diff(energy_envelope_high_db, prepend=energy_envelope_high_db[0])
    odf_high_db = np.maximum(odf_high_db, 0)  # half-wave rectifier

    # ODFs into a single array
    odf = np.vstack((odf_low_db, odf_high_db)).T

    numFrames = len(mX[:, 0])
    frmTime = H * np.arange(numFrames) / float(fs)

    maxplotfreq = 7000
    binFreq = fs * np.arange(N * maxplotfreq / fs) / N
    plt.subplot(2, 1, 1)
    plt.title("Original Signal STFT Spectrogram")
    plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:, :N * maxplotfreq // fs + 1]))
    plt.ylabel('Frequency (Hz)')

    plt.subplot(2, 1, 2)
    plt.plot(frmTime, odf_low_db, color="blue", label="Low Frequency ODF")
    plt.plot(frmTime, odf_high_db, color="green", label="High Frequency ODF")
    plt.legend()
    plt.title('Onset Detection Functions')
    plt.xlabel('Time (s)')
    plt.ylabel('ODF (dB)')
    plt.tight_layout()

    plt.savefig('ODFs.png')
    plt.show()

    return odf

computeODF('..\\..\\sounds\\piano.wav', 'blackman', 513, 1024, H=128)