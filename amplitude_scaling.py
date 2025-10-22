import numpy as np
import das4whales as dw
import matplotlib.pyplot as plt

"""
This file is written test any scaling artifacts from applying an F-K mask.
First, I want to create a narrowband signal propagating at a uniform speed, 
so the amplitude returned after masking is nearly identical to the original amplitude.
"""
# set up signal params:
fs = 200
dx = 8
nx = 100
ns = 800
c = 1500

cmin = 1400
cmax = 3000

t = np.arange(0, ns)/fs
t0 = 1

x = np.arange(0, nx)*dx

# build a sinc function:
fc = 20 # 20 Hz signal
BW = 5
sig = np.sinc(BW*(t-t0))*np.cos(2*np.pi*fc*(t-t0))

# test to make sure it worked:
S = np.fft.fft(sig)
f = np.fft.fftfreq(len(sig), 1/fs)

# plt.figure()
# plt.subplot(1,2,1)
# plt.plot(t, sig)
# plt.xlabel('time [s]')
# plt.ylabel('amplitude')

# plt.subplot(1,2,2)
# plt.plot(f, 20*np.log10(np.abs(S)))
# plt.xlabel('frequency [Hz]')
# plt.ylabel('magnitude')
# plt.xlim([0, fs/2])
# plt.show()

tx = np.zeros([nx, ns])

# make a T-X plot
for ix in range(nx):
    dt = ix*dx/c + t0
    tx[ix, :] = np.sinc(BW*(t-dt))*np.cos(2*np.pi*fc*(t-dt))

# make F-K filter and apply to data:
hyb_filt_sparse = dw.dsp.hybrid_ninf_filter_design(tx.shape, [0, nx-1, 1], dx, fs,
                                               cs_min=cmin, cp_min=cmin, 
                                               cp_max=cmax, cs_max=cmax,
                                               fmin=fc-10, fmax=fc+10., display_filter=False)
hyb_filt = hyb_filt_sparse.todense()

# apply filter
fk = np.fft.fftshift(np.fft.fft2(tx))
fk_filt = fk * hyb_filt
tx_filt = np.fft.ifft2(np.fft.ifftshift(fk_filt))

tx_filt_dw_sparse = dw.dsp.fk_filter_sparsefilt(tx, hyb_filt_sparse)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(tx, aspect='auto', extent=[t[0], t[-1], x[0], x[-1]])
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.colorbar()
plt.title('original t-x data')

plt.subplot(1,3,2)
plt.imshow(tx_filt.real, aspect='auto', extent=[t[0], t[-1], x[0], x[-1]])
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.colorbar()
plt.title('f-k filtered t-x data (eric''s version)')

plt.subplot(1,3,3)
plt.imshow(tx_filt_dw_sparse, aspect='auto', extent=[t[0], t[-1], x[0], x[-1]])
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.colorbar()
plt.title('f-k filtered t-x data (DAS4Whales''s version)')

# plot mask symmetry error:
M = hyb_filt
Mc = np.conj(np.flipud(np.fliplr(M)))
err = np.abs(M-Mc)

plt.figure()
plt.imshow(err, aspect='auto')
plt.colorbar()


mask_tx = np.fft.ifft2(np.fft.ifftshift(hyb_filt))
plt.figure()
plt.subplot(1,2,1)
plt.imshow(mask_tx.real, aspect='auto', extent=[t[0], t[-1], x[0], x[-1]])
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.colorbar()
plt.title('ifft of mask (real)')

plt.subplot(1,2,2)
plt.imshow(mask_tx.imag, aspect='auto', extent=[t[0], t[-1], x[0], x[-1]])
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.colorbar()
plt.title('ifft of mask (imag)')


# Try making and applying my own mask w/o fftshift:
mask = np.zeros(tx.shape).astype(complex)
freq = np.fft.fftfreq(ns, d=1/fs)
knum = np.fft.fftfreq(nx, d=dx)

for i in range(len(knum)):
    fmin = np.abs(knum[i]*cmin)
    fmax = np.abs(knum[i]*cmax)

    idx = np.where(np.logical_and(np.abs(f)>=fmin, np.abs(f)<=fmax))
    mask[i, idx] = 1

dwmask = dw.dsp.fk_filter_design(tx.shape, [0, nx, 1], dx, fs, cs_min=1400, cp_min=1400, cp_max=3000, cs_max=3000, display_filter=False)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(np.fft.fftshift(np.abs(mask)), aspect='auto', extent=[f[0], f[-1], knum[0], knum[-1]])
plt.xlabel('Freq [Hz]')
plt.ylabel('Wavenumber [1/m]')
plt.colorbar()
plt.title('my mask')

plt.subplot(1,3,2)
plt.imshow(np.abs(dwmask), aspect='auto', extent=[f[0], f[-1], knum[0], knum[-1]])
plt.xlabel('Freq [Hz]')
plt.ylabel('Wavenumber [1/m]')
plt.colorbar()
plt.title('DAS4Whales mask')

plt.subplot(1,3,3)
plt.imshow(np.fft.fftshift(np.abs(mask)) - np.abs(dwmask), aspect='auto', extent=[f[0], f[-1], knum[0], knum[-1]])
plt.xlabel('Freq [Hz]')
plt.ylabel('Wavenumber [1/m]')
plt.colorbar()
plt.title('difference')

# which masks have imaginary non-zero values?
mask_tx = np.fft.ifft2(mask)
dwmask_tx = np.fft.ifft2(np.fft.ifftshift(dwmask))

plt.figure()
plt.subplot(1,2,1)
plt.imshow(mask_tx.imag, aspect='auto', extent=[t[0], t[-1], x[0], x[-1]])
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.colorbar()
plt.title('imaginary component of my mask')

plt.subplot(1,2,2)
plt.imshow(dwmask_tx.imag, aspect='auto', extent=[t[0], t[-1], x[0], x[-1]])
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.colorbar()
plt.title('imaginary component of DAS4Whales mask')

# how does it compare when applied to data?
tx_filt_mymask = np.fft.ifft2(np.fft.fft2(tx)*mask)
tx_filt_dwmask = dw.dsp.fk_filter_filt(tx, dwmask)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(tx_filt_mymask.real, aspect='auto', extent=[t[0], t[-1], x[0], x[-1]])
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.colorbar()
plt.title('filtered signal (my mask)')

plt.subplot(1,3,2)
plt.imshow(tx_filt_dwmask, aspect='auto', extent=[t[0], t[-1], x[0], x[-1]])
plt.xlabel('Time (s)')
#plt.ylabel('Distance (m)')
plt.colorbar()
plt.title('filtered signal (DAS4Whales mask)')

plt.subplot(1,3,3)
plt.imshow(tx_filt_mymask.real-tx_filt_dwmask, aspect='auto', extent=[t[0], t[-1], x[0], x[-1]])
plt.xlabel('Time (s)')
#plt.ylabel('Distance (m)')
plt.colorbar()
plt.title('difference')


# test masking only on positive frequencies:
idx_half = np.arange(0, ns/2+1).astype(int)
half_mask = mask[:, :int(ns/2+1)]
half_freq = np.fft.rfftfreq(ns, 1/fs)
half_mask_iter = np.zeros_like(half_mask)
for i in range(len(knum)):
    fmin = np.abs(knum[i]*cmin)
    fmax = np.abs(knum[i]*cmax)

    idx = np.where(np.logical_and(half_freq>=fmin, half_freq<=fmax))
    half_mask_iter[i, idx] = 1 # produces identical results to half_mask

tmp = np.fft.fft2(tx)
half_fk = tmp[:, :ns//2+1]
half_fk_rfft = np.fft.fft(np.fft.rfft(tx, axis=1), axis=0)
half_fk_rfft2 =np.fft.rfft2(tx) # identical to above statement!

tx_half = np.fft.irfft2(half_fk_rfft2*half_mask)


plt.figure()
plt.subplot(1,3,1)
plt.imshow(tx_filt_mymask.real, aspect='auto', extent=[t[0], t[-1], x[0], x[-1]])
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.colorbar()
plt.title('filtered signal (my mask)')

plt.subplot(1,3,2)
plt.imshow(tx_half, aspect='auto', extent=[t[0], t[-1], x[0], x[-1]])
plt.xlabel('Time (s)')
#plt.ylabel('Distance (m)')
plt.colorbar()
plt.title('filtered signal (positive frequencies mask)')

plt.subplot(1,3,3)
plt.imshow(tx_filt_mymask.real-tx_half, aspect='auto', extent=[t[0], t[-1], x[0], x[-1]])
plt.xlabel('Time (s)')
#plt.ylabel('Distance (m)')
plt.colorbar()
plt.title('difference')

plt.show()