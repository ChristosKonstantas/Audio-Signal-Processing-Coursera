import numpy as np
import sys, os, math
import scipy.signal as scsg
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..\\..\\software\\models\\'))
import utilFunctions as UF
import dftModel as DFT
from scipy.fft import ifft, fftshift
import sineModel as SM
eps = np.finfo(float).eps
import stft as STFT
import harmonicModel as HM


def sineModelMultiRes(x, fs, W, N, t, B):
    """
    Analysis/synthesis of a sound using the sinusoidal model with multi-resolution analysis
    x: input array sound, fs: sampling rate, W: list of analysis windows,
    N: list of FFT sizes, t: threshold in negative dB, B: list of frequency bands
    returns y: output array sound
    """
    # Define half window sizes for the three windows
    hM1 = [int(np.floor((W[i].size + 1) / 2)) for i in range(len(W))]
    hM2 = [int(np.floor(W[i].size / 2)) for i in range(len(W))]

    Ns = 512  # FFT size for synthesis (even)
    H = Ns // 4  # Hop size used for analysis and synthesis
    hNs = Ns // 2  # half of synthesis FFT size
    pin = [max(hNs, hM1[i]) for i in range(len(W))]  # init sound pointer in middle of anal window
    pend = [x.size - max(hNs, hM1[i]) for i in range(len(W))]  # last sample to start a frame
    yw = np.zeros(Ns)  # initialize output sound frame
    y = np.zeros(x.size)  # initialize output array
    # Normalize analysis windows
    for i in range(len(W)):
        W[i] = W[i] / sum(W[i])
    sw = np.zeros(Ns)  # initialize synthesis window
    ow = scsg.get_window('triang', (2 * H), True)  # triangular window
    sw[hNs - H:hNs + H] = ow  # add triangular window
    bh = scsg.get_window('blackmanharris', Ns, Ns%2==0)  # blackmanharris window
    bh = bh / sum(bh)  # normalized blackmanharris window
    sw[hNs - H:hNs + H] = sw[hNs - H:hNs + H] / bh[hNs - H:hNs + H]  # normalized synthesis window
    flag_f0 = False
    for i in range(len(W)):
        while pin[i] < pend[i]:  # while input sound pointer is within sound
            # print(pin[i],i)
            # -----analysis-----
            x1 = x[pin[i] - hM1[i]:pin[i] + hM2[i]]  # select frame
            mX, pX = DFT.dftAnal(x1, W[i], N[i])  # compute dft

            # Compute the frequency bin indices for the specified bands
            ploc = UF.peakDetection(mX, t)  # detect locations of peaks
            iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)  # refine peak values by interpolation
            ipfreq = fs * iploc / float(N[i])  # convert peak locations to Hertz
            k = np.nonzero((ipfreq >= B[i][0]) & (ipfreq < B[i][1]))[0]
            # if not flag_f0:
            #     f0et = 2
            #     minf0 = 20
            #     maxf0 = 100
            #     f0 = UF.f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, 0)  # best candidate f0
            #     flag_f0 = True
            # harmDevSlope = 0.01
            # hfreq, hmag, hphase = HM.harmonicDetection(ipfreq[k], ipmag[k], ipphase[k], f0, int(22050//f0), [], fs, harmDevSlope)

            # -----synthesis-----
            Y = UF.genSpecSines(ipfreq[k], ipmag[k], ipphase[k], Ns, fs)  # generate sines in the spectrum
            fftbuffer = np.real(ifft(Y))  # compute inverse FFT
            yw[:hNs - 1] = fftbuffer[hNs + 1:]  # undo zero-phase window
            yw[hNs - 1:] = fftbuffer[:hNs + 1]
            y[pin[i] - hNs:pin[i] + hNs] += sw * yw  # overlap-add and apply a synthesis window
            pin[i] += H  # advance sound pointer
    return y


# Define windows
w1 = scsg.get_window('blackman', 4095, False)
w2 = scsg.get_window('blackman', 2047, False)
w3 = scsg.get_window('hamming', 1023, False)


# Define FFT sizes
N1 = 8192
N2 = 4096
N3 = 2048

# Define frequency bands
B1 = (0, 1000)
B2 = (1000, 5000)
B3 = (5000, 22050)

# Example sound input and parameters
fs, x = UF.wavread('..\\..\\sounds\\orchestra.wav')
t = -120  # Threshold in dB

# Call the multi-resolution sine model function
y = sineModelMultiRes(x, fs, [w1, w2, w3], [N1, N2, N3], t, [B1, B2, B3])

plt.subplot(2,1,1)
mX, _ = STFT.stftAnal(x, scsg.get_window('blackman', 2047), 4096, 200)
plt.pcolormesh(np.transpose(mX))
plt.subplot(2,1,2)
mY, _ = STFT.stftAnal(y, scsg.get_window('blackman', 2047), 4096, 200)
plt.pcolormesh(np.transpose(mY))
plt.show()


plt.title("Original Signal")
plt.plot(x)
plt.show()

# Plot synthesized signal
plt.title("Synthesized Signal")
plt.plot(y)
plt.show()

H = 200
N = 4096
maxplotfreq = 22049
numFrames = int(mX[:,0].size)
frmTime = H * np.arange(numFrames)/float(fs)
binFreq = fs*np.arange(N*maxplotfreq/fs)/N
plt.title("Original Signal STFT Spectrogram")
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq//fs + 1]))
plt.show()

numFrames = int(mY[:,0].size)
frmTime = H * np.arange(numFrames)/float(fs)
binFreq = fs*np.arange(N*maxplotfreq/fs)/N
plt.title("Synthesized Signal STFT Spectrogram")
plt.pcolormesh(frmTime, binFreq, np.transpose(mY[:,:N*maxplotfreq//fs + 1]))
plt.show()
UF.wavwrite(y, fs, filename='orchestra_sinusoidal.wav')

