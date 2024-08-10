import numpy as np
import sys, os, math
import scipy.signal as scsg
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..\\..\\software\\models//'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..\\..\\software\\transformations//'))
import utilFunctions as UF
import matplotlib.pyplot as plt
import dftModel as DFT
import sineModel as SM

"""
Part 1 (stftTransformations.py stftFiltering)
"""

fs, x = UF.wavread('..\\..\\sounds\\oboe-A4.wav')

M = N = 512
w = scsg.get_window('hann', M, M%2==0)
xw = x[10000:10000+M]*w
filter = scsg.get_window('hamming', 30, 30%2==0) * -60 # filter in frequency response
mX, pX = DFT.dftAnal(xw, w, N)

centerbin = 40
mY = np.copy(mX)
mY[centerbin-15:centerbin+15] = mX[centerbin-15:centerbin+15] + filter

y= DFT.dftSynth(mY, pX, N) * sum(w)

plt.plot(mX)
plt.plot(mY)
plt.show()

""" 
Morphing (stftTransformations.py stftMorph)
"""

fs, x1 = UF.wavread('..\\..\\sounds\\rain.wav')
fs, x2 = UF.wavread('..\\..\\sounds\\soprano-E4.wav')

M = N = 512
w = scsg.get_window('hann', M, M%2==0)
x1w = x1[10000:10000+M]*w
x2w = x2[10000:10000+M]*w

mX1, pX1 = DFT.dftAnal(x1w, w, N)
mX2, pX2 = DFT.dftAnal(x2w, w, N)

smoothf = .2
mX2smooth1 = scsg.resample(np.maximum(-200,mX2), int(mX2.size*smoothf))
mX2smooth2 = scsg.resample(mX2smooth1, N//2+1)

balancef = 0.7
mY = balancef * mX2smooth2 + (1-balancef) * mX1

y = DFT.dftSynth(mY, pX1, N)

plt.plot(mX)
plt.plot(mY)
plt.show()


"""
Part 3
"""

inputFile ='..\\..\\sounds\\piano.wav'
window = 'hamming'
M = 1001
N = 2048
t = -100
minSineDur = 0.01
maxnSines = 150
freqDevOffset = 30
freqDevSlope = 0.02

Ns = 512
H = 128

fs,x = UF.wavread(inputFile)

w = scsg.get_window(window, M, M % 2 == 0)

tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)


y = SM.sineModelSynth(tfreq, tmag, tphase, Ns, H, fs)  # try also with tphase = np.array([])

UF.wavwrite(y, fs, 'test2.wav')