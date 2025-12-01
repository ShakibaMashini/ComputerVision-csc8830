import cv2
import numpy as np


def load_color_image(path: str) -> np.ndarray:
    """
    Load an image from disk, convert it to RGB, and normalize pixel values to [0, 1].
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Image not found: " + path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def create_gaussian_kernel(ksize=51, sigma=12.0):
    """
    Create a normalized 2D Gaussian kernel of size ksize x ksize with standard deviation sigma.
    Used as the blur point-spread function (PSF).
    """
    g = cv2.getGaussianKernel(ksize, sigma)
    kernel = g @ g.T
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def psf_to_otf(psf, shape):
    """
    Convert a spatial PSF (blur kernel) into its frequency-domain representation (OTF).
    The PSF is padded and circularly shifted so that the FFT behaves like convolution.
    """
    H, W = shape
    h, w = psf.shape

    padded = np.zeros((H, W), dtype=np.float32)
    cy, cx = h // 2, w // 2

    padded[:h-cy, :w-cx] = psf[cy:, cx:]
    padded[:h-cy, W-cx:] = psf[cy:, :cx]
    padded[H-cy:, :w-cx] = psf[:cy, cx:]
    padded[H-cy:, W-cx:] = psf[:cy, :cx]

    return np.fft.fft2(padded)


def wiener_filter(blurred: np.ndarray, kernel: np.ndarray, K=0.002):
    """
    Restore a blurred RGB image using Wiener deconvolution.
    The parameter K controls how strongly noise is suppressed.
    """
    restored = np.zeros_like(blurred)

    for c in range(3):
        channel = blurred[:, :, c]
        G = np.fft.fft2(channel)

        H = psf_to_otf(kernel, channel.shape)
        H_conj = np.conjugate(H)

        F = (H_conj / (H * H_conj + K)) * G

        f = np.fft.ifft2(F)
        restored[:, :, c] = np.real(f)

    return np.clip(restored, 0, 1)


def main():
    """
    Load the original image, blur it using a Gaussian filter, restore it in the frequency domain,
    and save the blurred and restored results.
    """
    img = load_color_image("part2/Nailpolish.jpeg")
    kernel = create_gaussian_kernel(ksize=51, sigma=15.0)

    blurred = np.zeros_like(img)
    for c in range(3):
        blurred[:, :, c] = cv2.filter2D(img[:, :, c], -1, kernel)

    restored = wiener_filter(blurred, kernel, K=0.001)

    cv2.imwrite("part2/Nailpolish_blurred.png", (blurred[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite("part2/Nailpolish_restored.png", (restored[:, :, ::-1] * 255).astype(np.uint8))
    cv2.imwrite("part2/Nailpolish_original.png", (img[:, :, ::-1] * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
