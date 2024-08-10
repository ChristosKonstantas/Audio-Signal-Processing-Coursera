import numpy as np
from scipy.fftpack import fft, ifft
import sys, os, math
import scipy.signal as scsg
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..\\..\\software\\models\\'))
import utilFunctions as UF
import stochasticModel as STM
import matplotlib.pyplot as plt
import dftModel as DFT
import harmonicModel as HM


fs, x = UF.wavread('..\\..\\sounds\\ocean.wav')
M = N = 256
stocf = 0.2  # smoothing factor / downsampling factor (we want to reduce info by a factor) (=1 then lossless)
w = scsg.get_window('hann', M, M % 2 == 0)
xw = x[1000:1000 + M] * w
X = fft(xw)
mX = 20 * np.log10(abs(X[:int(N / 2)]) + 1e-16)
# downsampling by a factor of stocf (0.2 = 1/5 of the samples we started with)
mXenv = scsg.resample(np.maximum(-200, mX), int(N / 2 * stocf))
plt.plot(xw)
plt.show()
plt.plot(np.arange(mXenv.size) / stocf, mXenv)
plt.plot(mX)
plt.show()

mY = scsg.resample(mXenv, N // 2)
pY = 2 * np.pi * np.random.rand(N // 2)  # generate random phase
Y = np.zeros(N, dtype=complex)
Y[:N // 2] = 10 ** (mY / 20) * np.exp(1j * pY)
Y[N // 2 + 1:] = 10 ** (mY[:0:-1] / 20) * np.exp(-1j * pY[:0:-1])
y = np.real(ifft(Y))
plt.plot(xw)
plt.plot(y)
# signals in time domain do not show the exact reconstruction but what matters is the magnitude spectrum distribution
plt.show()

H = 128
stocf = 0.2
stocEnv = STM.stochasticModelAnal(x, H, H * 2, stocf)
# stocEnv (frames, approximation_points)
plt.pcolormesh(np.transpose(stocEnv))
plt.show()

"""
Part 2 hprModel, hpsModel
"""

# subtraction of sinusoids from original signal

pin = 40000
M = 801
t = -80
minf0 = 300
maxf0 = 500
f0et = 5
nH = 60
harmDevSlope = .001
N = 2048
w = scsg.get_window('blackman', M, M%2==0)
hM1 = int(np.floor((M+1)/2))
hM2 = int(np.floor(M/2))

x1 = x[pin-hM1:pin+hM2] #center of the window
mX, pX = DFT.dftAnal(x1, w, N)
ploc = UF.peakDetection(mX, t)
iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)
ipfreq = fs * iploc / N # locations to hz
f0 = UF.f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, 0) # best candidate f0
hfreq, hmag, hphase = HM.harmonicDetection(ipfreq, ipmag, ipphase, f0, nH, [], fs, harmDevSlope)


# way 1 (hpr)
Ns = 512
hNs = Ns // 2
Yh = UF.genSpecSines(hfreq, hmag, hphase, Ns, fs) # synthesize spectral sines (main lobes of window)
wr = scsg.get_window('blackmanharris', Ns)
xw2 = x[pin-hNs-1:pin+hNs-1] * wr / sum(wr)
fftbuffer = np.zeros(Ns)
fftbuffer[:hNs] = xw2[hNs:]
fftbuffer[hNs:] = xw2[:hNs]
X2 = fft(fftbuffer)
Xr = X2 - Yh
plt.title('hpr')
plt.plot(X2)
plt.plot(Yh)
plt.show()


# way 2 (hps)
stocf = 0.4
mXr = 20*np.log10(abs(Xr[:hNs]))
mXrenv = scsg.resample(np.maximum(-200, mXr), int(mXr.size*stocf))
stocEnv = scsg.resample(mXrenv, hNs)
plt.title('hps')
plt.plot(mXr)
plt.plot(stocEnv)
plt.show()
