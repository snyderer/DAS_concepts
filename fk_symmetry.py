import numpy as np
import matplotlib.pyplot as plt

fs = 200
dx = 8
nx = int(40000/dx)
ns = int(fs*30)
c = 1500  # m/s

tx = np.zeros((nx, ns)) # initialize space-time array
t = np.arange(ns)/fs
tsrc = 5  # source time in s
dist = np.arange(nx)*dx
dist_src = 10000  # source location along cable in m
dist_bend = 25000  # location of bend in cable in m
bend_angle_deg = 60  # angle of bend in degrees
bend_angle_rad = bend_angle_deg * np.pi / 180

c = 1500

# convert cable segments to x,y coordinates with source at (0,0)
x = np.zeros(nx)
y = np.zeros(nx)
for ix in range(1, nx):
    if dist[ix] <= dist_bend:
        x[ix] = dist[ix] - dist_src
        y[ix] = 0
    else:
        x[ix] = (dist_bend - dist_src) + (dist[ix] - dist_bend)*np.cos(bend_angle_rad)
        y[ix] = (dist[ix] - dist_bend)*np.sin(bend_angle_rad)
    # add delta functions along arrival path 
    r = np.sqrt(x[ix]**2 + y[ix]**2)
    t_arrival = r / c
    it_arrival = int(np.round(t_arrival * fs)) + int(np.round(tsrc * fs))
    if it_arrival < ns:
        tx[ix, it_arrival] = 1  # delta function at arrival time


# zero-out one quarter to break x dimension symmetry
#tx[:nx//4, :] = 0

# simulate a linear chirp signal
f0 = 10  # starting frequency
f1 = 50  # ending frequency
sig_duration = 2  # seconds
nsig = int(sig_duration*fs)
signal = np.sin(2*np.pi*(f0*t[0:nsig] + (f1-f0)/(2*10)*t[0:nsig]**2))

plt.figure()
plt.plot(t[0:nsig], signal)

# convolve the delta functions with the signal
for ix in range(nx):
    tx[ix, :] = np.convolve(tx[ix, :], signal, mode='same')

plt.figure()
plt.subplot(1,3,1)
plt.imshow(tx, extent=[t[0], t[-1], dist[-1]/1000, dist[0]/1000], aspect='auto', vmin=-0.1, vmax=0.1, cmap='seismic')
plt.xlabel('Time [s]') 
plt.ylabel('Distance along cable [km]')
plt.title('T-X simulation')
plt.colorbar(label='Amplitude')

# FFT to F-K domain
D = np.fft.fftshift(np.fft.fft2(tx))
Ddb = 20*np.log10(np.abs(D)/np.max(np.abs(D)))
nk, nt = D.shape
dt = 1/fs
k = np.fft.fftshift(np.fft.fftfreq(nk, d=dx))
f = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))
plt.subplot(1,3,2)
plt.imshow(Ddb, extent=[f[0], f[-1], k[0], k[-1]], aspect='auto', vmin=np.min(Ddb), vmax=np.max(Ddb)/10, cmap='viridis')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Wavenumber [1/m]')
plt.title('F-K spectrum')

plt.subplot(1,3,3)
plt.imshow(np.angle(D), extent=[f[0], f[-1], k[0], k[-1]], aspect='auto', vmin=-np.pi, vmax=np.pi, cmap='viridis')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Wavenumber [1/m]')
plt.colorbar(label='Phase [radians]')
plt.title('F-K phase')

plt.show()
