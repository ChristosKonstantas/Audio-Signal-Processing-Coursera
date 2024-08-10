import numpy as np
import math
import matplotlib.pyplot as plt
eps = np.finfo(float).eps
import scipy.signal as scsg
from scipy.fftpack import fft, fftshift

""" 
A4-Part-1: Extracting the main lobe of the spectrum of a window

Write a function that extracts the main lobe of the magnitude spectrum of a window given a window 
type and its length (M). The function should return the samples corresponding to the main lobe in 
decibels (dB).

To compute the spectrum, take the FFT size (N) to be 8 times the window length (N = 8*M) (For this 
part, N need not be a power of 2). 

The input arguments to the function are the window type (window) and the length of the window (M). 
The function should return a numpy array containing the samples corresponding to the main lobe of 
the window. 

In the returned numpy array you should include the samples corresponding to both the local minimas
across the main lobe. 

The possible window types that you can expect as input are rectangular ('boxcar'), 'hamming' or
'blackmanharris'.

NOTE: You can approach this question in two ways: 1) You can write code to find the indices of the 
local minimas across the main lobe. 2) You can manually note down the indices of these local minimas 
by plotting and a visual inspection of the spectrum of the window. If done manually, the indices 
have to be obtained for each possible window types separately (as they differ across different 
window types).

Tip: log10(0) is not well defined, so its a common practice to add a small value such as eps = 1e-16 
to the magnitude spectrum before computing it in dB. This is optional and will not affect your answers. 
If you find it difficult to concatenate the two halves of the main lobe, you can first center the 
spectrum using fftshift() and then compute the indexes of the minimas around the main lobe.


Test case 1: If you run your code using window = 'blackmanharris' and M = 100, the output numpy 
array should contain 65 samples.

Test case 2: If you run your code using window = 'boxcar' and M = 120, the output numpy array 
should contain 17 samples.

Test case 3: If you run your code using window = 'hamming' and M = 256, the output numpy array 
should contain 33 samples.

"""
def extractMainLobe(window, M):
    """
    Input:
            window (string): Window type to be used (Either rectangular ('boxcar'), 'hamming' or '
                blackmanharris')
            M (integer): length of the window to be used
    Output:
            The function should return a numpy array containing the main lobe of the magnitude 
            spectrum of the window in decibels (dB).
    """

    w = scsg.windows.get_window(window, Nx=M, fftbins=M % 2 == 0)

    # Calculate the middle of the window
    hM1 = int(math.floor((M + 1) / 2))
    hM2 = int(math.floor(M / 2))

    # FFT buffer length
    # Find the next power of two greater than or equal to N
    N = 8 * M
    hN = N // 2

    # Initialize FFT buffer
    fftbuffer = np.zeros(N)

    # Place window around the 0th sample
    fftbuffer[:hM1] = w[hM2:]
    fftbuffer[N - hM2:] = w[:hM2]
    # Compute the spectrum of the buffer using FFT

    W_f = fftshift(fft(fftbuffer))
    # Convert complex spectrum into absolute value and phase
    absX = abs(W_f)
    absX[absX < np.finfo(float).eps] = np.finfo(float).eps  # Prevent log(0) issues
    mX = 20 * np.log10(absX + 1e-16)  # Conversion to decibels
    mX1 = np.zeros(N)
    mX1[:hN] = mX[hN:]
    mX1[N - hN:] = mX[:hN]
    prev = mX1[0]
    main_lobe_samples_left = []
    main_lobe_samples_left.append(prev)
    for i in range(1, len(mX1)):
        if mX1[i] > prev:
            break
        else:
            main_lobe_samples_left.append(mX1[i])
        prev = mX1[i]

    main_lobe_samples_right = [t for t in reversed(main_lobe_samples_left) if t != mX1[0]]
    main_lobe_samples = main_lobe_samples_right + main_lobe_samples_left

    return np.array(main_lobe_samples)