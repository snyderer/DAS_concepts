import das4whales as dw
import tapering_fk_filters as tap
import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 200       # Hz
dx = 8.0        # m
nt, nx = int(2*fs), int(800/dx)
c = 1500 # m/s
cs_min, cp_min, cp_max, cs_max = 1400, 1450, 3400, 3500

# Step 1: Make impulse response for array
ir = tap.make_impulse_response(nt, nx, fs, dx, c)

# Step 2: Define signal of interest (Gaussian-modulated sinusoid)
t = np.arange(0, 1, 1/fs)
f0 = 40  # Hz
signal = np.sin(2 * np.pi * f0 * t) * np.exp(-((t - 0.1)**2) / (2 * (0.01)**2))
# signal = np.hanning(int(.1*fs))*np.random.randn(int(.1*fs))

# Step 3: Convolve signal with impulse response to get array data
tx = tap.convolve_signal_with_ir(signal, ir)
tx = tx[:nt, :]  # truncate to original length

noise_level = 0
tx += noise_level*np.random.randn(tx.shape[0], tx.shape[1]) # add noise

# Step 4: Create masks
# def fk_filter_design(trace_shape, selected_channels, dx, fs, cs_min=1400, cp_min=1450, cp_max=3400, cs_max=3500, display_filter=False):

mask_binary = dw.dsp.fk_filter_design(tx.shape, [0, tx.shape[0], 1], dx, fs,
                                        cs_min=cs_min, cp_min=cp_min, cp_max=cp_max, cs_max=cs_max)

mask_smooth = tap.smooth_mask(mask_binary, [5,5])

# hyb_filt_sparse = dw.dsp.hybrid_ninf_filter_design(tx.shape, [0, nx-1, 1], dx, fs,
#                                                cs_min=cs_min, cp_min=cp_min, 
#                                                cp_max=cp_max, cs_max=cs_max,
#                                                fmin=f0-30, fmax=f0+30., display_filter=False)
fk_params = {}
fk_params['fmin'] = f0-30
fk_params['fmax'] = f0+30
fk_params['c_min'] = cs_min
fk_params['c_max'] = cp_max

hyb_filt_sparse = dw.dsp.hybrid_ninf_gs_filter_design(tx.shape, [0, nx-1, 1], dx, fs, 
                                                      fk_params, display_filter=False)

# convert sparse matrix to regular matrix
hyb_filt = hyb_filt_sparse.todense()

# plot masks:
fig1, axm = plt.subplots(1,4, figsize=(18,8), sharex=True, sharey=True)
axm[0].imshow(mask_binary, extent=[-fs/2, fs/2, 0, fs/c], aspect='auto')
axm[0].set_xlabel('Freq [Hz]')
axm[0].set_ylabel('k [1/m]')
axm[0].set_title('mask w/o tapering')

axm[1].imshow(mask_smooth, extent=[-fs/2, fs/2, 0, fs/c], aspect='auto')
axm[1].set_xlabel('Freq [Hz]')
axm[1].set_ylabel('k [1/m]')
axm[1].set_title('mask w tapering')

axm[2].imshow(hyb_filt, extent=[-fs/2, fs/2, 0, fs/c], aspect='auto')
axm[2].set_xlabel('Freq [Hz]')
axm[2].set_ylabel('k [1/m]')
axm[2].set_title('D4W Hybrid Filter')

# Apply filters to data
tx_binMask = dw.dsp.fk_filter_filt(tx, mask_binary, True)
tx_smoothMask = dw.dsp.fk_filter_filt(tx, mask_smooth, True)
tx_hyb = dw.dsp.fk_filter_sparsefilt(tx, hyb_filt_sparse, True)

# plot masked data:
fig1, axm = plt.subplots(1,4, figsize=(18,8), sharex=True, sharey=True)
axm[0].imshow(tx, extent=[0, nt/fs, 0, nx*dx], aspect='auto')
axm[0].set_xlabel('time [s]')
axm[0].set_ylabel('span [m]')
axm[0].set_title('original data')


axm[1].imshow(tx_binMask, extent=[0, nt/fs, 0, nx*dx], aspect='auto')
axm[1].set_xlabel('time [s]')
axm[1].set_ylabel('span [m]')
axm[1].set_title('data w/o tapered mask')

axm[2].imshow(tx_smoothMask, extent=[0, nt/fs, 0, nx*dx], aspect='auto')
axm[2].set_xlabel('time [s]')
axm[2].set_ylabel('span [m]')
axm[2].set_title('data w/ tapered mask')

axm[3].imshow(tx_hyb, extent=[0, nt/fs, 0, nx*dx], aspect='auto')
axm[3].set_xlabel('time [s]')
axm[3].set_ylabel('span [m]')
axm[3].set_title('data w/ hybrid mask from DAS4Whales')

# plot error data:
tx = tx/tx.max()
fig1, axm = plt.subplots(1,3, figsize=(18,8), sharex=True, sharey=True)
axm[0].imshow(np.abs(tx_binMask/tx_binMask.max()-tx), extent=[0, nt/fs, 0, nx*dx], aspect='auto')
axm[0].set_xlabel('time [s]')
axm[0].set_ylabel('span [m]')
axm[0].set_title('error w/o tapered mask')

axm[1].imshow(np.abs(tx_smoothMask/tx_smoothMask.max()-tx), extent=[0, nt/fs, 0, nx*dx], aspect='auto')
axm[1].set_xlabel('time [s]')
axm[1].set_ylabel('span [m]')
axm[1].set_title('error w/ tapered mask')

im = axm[2].imshow(np.abs(tx_hyb/tx_hyb.max()-tx), extent=[0, nt/fs, 0, nx*dx], aspect='auto')
axm[2].set_xlabel('time [s]')
axm[2].set_ylabel('span [m]')
axm[2].set_title('error w/ hybrid mask from DAS4Whales')
for i in [0,2]:
    fig1.colorbar(im, ax=axm[i])

plt.show()