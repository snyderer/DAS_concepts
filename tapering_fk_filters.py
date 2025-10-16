import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from scipy.signal import fftconvolve
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
import das4whales as dw

# --------------------------
# 1. Impulse response generator
# --------------------------
def make_impulse_response(nt, nx, fs, dx, velocity, delay=0.0):
    """
    Generate an array impulse response for a plane wave with given velocity.
    Each sensor (x) receives an impulse delayed by x / velocity.
    """
    t = np.arange(nt) / fs
    x = np.arange(nx) * dx
    impulse_response = np.zeros((nt, nx))
    for i, xi in enumerate(x):
        arrival_time = xi / velocity + delay
        it = int(np.round(arrival_time * fs))
        if 0 <= it < nt:
            impulse_response[it, i] = 1.0
    return impulse_response

# --------------------------
# 2. Convolution function
# --------------------------
def convolve_signal_with_ir(signal, ir):
    """
    Convolve 1D signal (time) with 2D impulse response (time x sensors).
    Returns 2D array (time x sensors) of convolved signals.
    """
    nt, nx = ir.shape
    out = np.zeros((len(signal) + nt - 1, nx))
    for i in range(nx):
        out[:, i] = fftconvolve(signal, ir[:, i], mode='full')
    return out

# --------------------------
# 3. Mask and filtering helpers
# --------------------------
def make_velocity_mask(nt, nx, fs, dx, vel_min, vel_max):
    """Binary mask based on velocity corridor in F–K domain."""
    f = np.fft.fftshift(np.fft.fftfreq(nt, 1/fs))
    k = np.fft.fftshift(np.fft.fftfreq(nx, dx))
    F, K = np.meshgrid(f, k, indexing='ij')
    phase_vel = np.divide(F, K, out=np.zeros_like(F), where=K!=0)
    return (np.abs(phase_vel) >= vel_min) & (np.abs(phase_vel) <= vel_max)

def smooth_mask(mask_binary, smooth_px=6):
    """
    Smooth (apodize) a binary mask using a localized Gaussian blur,
    but preserve 1 and 0 far from the edges.
    """
    # Gaussian blur to soften edges
    blurred = gaussian_filter(mask_binary.astype(float), sigma=smooth_px)
    # Renormalize to 0–1
    blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min())
    # Apply a cosine ramp to make transition smoother
    transition = 0.5 - 0.5 * np.cos(np.pi * np.clip(blurred, 0, 1))
    return transition

def apply_fk_filter(tx, fs, dx, mask, fk_taper_alpha=None):
    """FFT → optional F–K taper → mask → inverse FFT."""
    FK = np.fft.fftshift(np.fft.fft2(tx))
    if fk_taper_alpha is not None:
        nt, nx = FK.shape
        w_f = tukey(nt, fk_taper_alpha)
        w_k = tukey(nx, fk_taper_alpha)
        FK *= np.outer(w_f, w_k)
    FKf = FK * mask
    tx_out = np.fft.ifft2(np.fft.ifftshift(FKf))
    return np.real(tx_out)

# --------------------------
# 4. Main simulation
# --------------------------
if __name__ == "__main__":
    # Parameters
    fs = 200       # Hz
    dx = 8.0        # m
    nt, nx = int(2*fs), int(800/dx)
    c = 1500 # m/s
    cs_min, cs_max, cp_min, cp_max = 1400, 5000, 1450, 5500

    # Step 1: Make impulse response for array
    ir = make_impulse_response(nt, nx, fs, dx, c)

    # Step 2: Define signal of interest (Gaussian-modulated sinusoid)
    t = np.arange(0, 1, 1/fs)
    f0 = 40  # Hz
    signal = np.sin(2 * np.pi * f0 * t) * np.exp(-((t - 0.1)**2) / (2 * (0.01)**2))
    # signal = np.hanning(int(.1*fs))*np.random.randn(int(.1*fs))

    # Step 3: Convolve signal with impulse response to get array data
    tx = convolve_signal_with_ir(signal, ir)
    tx = tx[:nt, :]  # truncate to original length

    tx += .001*np.random.randn(tx.shape[0], tx.shape[1]) # add noise
    
    # Step 4: Create masks
    # def fk_filter_design(trace_shape, selected_channels, dx, fs, cs_min=1400, cp_min=1450, cp_max=3400, cs_max=3500, display_filter=False):

    mask_binary = dw.dsp.fk_filter_design(tx.shape, [0, tx.shape[0], 1], dx, fs,
                                           cs_min=cs_min, cp_min=cp_min, cp_max=cp_max, cs_max=cs_max)
    mask_smooth = smooth_mask(mask_binary, smooth_px=10)

    fig1, axm = plt.subplots(1,2, figsize=(18,8), sharex=True, sharey=True)
    im1 = axm[0].imshow(mask_binary, extent=[-fs/2, fs/2, 0, fs/c], aspect='auto')
    axm[0].set_xlabel('Freq [Hz]')
    axm[0].set_ylabel('k [1/m]')
    axm[0].set_title('mask w/o tapering')

    im2 = axm[1].imshow(mask_smooth, extent=[-fs/2, fs/2, 0, fs/c], aspect='auto')
    axm[1].set_xlabel('Freq [Hz]')
    axm[1].set_ylabel('k [1/m]')
    axm[1].set_title('mask w tapering')
    
    # Step 5: Apply four test cases
    tx_orig = tx
    tx_fk_no_taper = apply_fk_filter(tx, fs, dx, mask_binary, fk_taper_alpha=None)
    tx_fk_taper_data = apply_fk_filter(tx, fs, dx, mask_binary, fk_taper_alpha=0.2)
    tx_fk_taper_both = apply_fk_filter(tx, fs, dx, mask_smooth, fk_taper_alpha=0.2)

    # --------------------------
    # Plot results (time = x-axis, distance = y-axis)
    # --------------------------
    extent = [0, nt/fs, nx*dx, 0]  # time (s) on x, distance (m) on y
    titles = [
        "1. Original convolved signal",
        "2. F–K filtered (no taper)",
        "3. F–K filtered (tapered edges F–K data)",
        "4. F–K filtered (tapered data + mask)"
    ]
    images = [tx_orig, tx_fk_no_taper, tx_fk_taper_data, tx_fk_taper_both]

    fig, axs = plt.subplots(1, 4, figsize=(18, 4), sharex=True, sharey=True)
    for ax, img, title in zip(axs, images, titles):
        im = ax.imshow(img.T, aspect='auto', cmap='seismic', extent=extent)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
    axs[0].set_ylabel("Distance (m)")
    fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.04, pad=0.1)
    plt.tight_layout()
    plt.show()
