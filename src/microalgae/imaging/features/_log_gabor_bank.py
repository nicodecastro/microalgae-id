"""Fixed-profile port of the LogGabor MATLAB toolbox cited by Sanchez (2019).

Original code: Sylvain Fischer (2007), published by Gabriel Cristobal,
Sylvain Fischer and Rafael Redondo (2016), under the GNU GPL (version not
specified by the upstream archive). This Python adaptation retains that
upstream GPL notice; see https://www.gnu.org/copyleft/gpl.html.
Source: https://doi.org/10.6084/m9.figshare.2067504.v1
Archive: https://ndownloader.figshare.com/files/3678759 (log_gabors.zip)
SHA256: 1815be259337b3ec3d81f5b35c4905d4a55b60e951a4edae14f170c0cb50b901

Ported paths: logGaborFilters.m, logKernel.m, polarcoor.m, symetriefft.m,
dcmp.m, and the unit-coefficient amplitude calibration in rcmp.m. Only the
256 x 256, four-scale/six-orientation, full-resolution profile is supported:
param = [0,.5,0,2,0,0,0,0,0,1,1,0,3,5,6,0,0,2,0,3,2,0,3], csf = -1.
The low-pass is retained for normalization only. Bandpass DC is explicitly
zeroed rather than retaining the source's tiny log(1e-5) approximation.
This is a chosen toolbox configuration, not verified Sanchez experiment parity.
"""

import numpy as np


def _validate_image(image):
    image = np.asarray(image)
    if image.ndim != 2 or not image.size or image.dtype.kind not in "buif":
        raise ValueError("image must be a nonempty 2-D real numeric array")
    if not np.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    return image.astype(np.float64, copy=False)


def _mirror_spectrum(spectrum, *, include_edges=True):
    """MATLAB symetriefft for an even, fftshift-ordered Fourier grid."""
    mirrored = np.conj(np.roll(spectrum[::-1, ::-1], (1, 1), axis=(0, 1)))
    if not include_edges:
        mirrored[0, :] = 0
        mirrored[:, 0] = 0
    return mirrored


def _polar_grid(size):
    coordinates = np.arange(size) - size // 2
    row, column = np.meshgrid(coordinates, coordinates, indexing="ij")
    radius = np.hypot(row, column)
    radius[size // 2, size // 2] = 1e-5
    angle = -np.arcsin(row / radius)
    alternate = angle + 2 * np.pi
    alternate[:, :size // 2] = np.pi - angle[:, :size // 2]
    angle[:, :size // 2] = -np.pi - angle[:, :size // 2]
    return radius, angle, alternate, row, column


def _log_kernel(grid, angle, frequency):
    radius, theta, alternate, *_ = grid
    angle = (angle + np.pi / 2) % (2 * np.pi) - np.pi / 2
    angular = np.minimum(
        np.abs(theta - angle) % (2 * np.pi),
        np.abs(alternate - angle) % (2 * np.pi),
    ) / (np.pi / 6)
    radial = np.log2(radius / frequency)
    return np.exp(-0.996 * (0.75 * radial ** 2 + angular ** 2))


def _channel(grids, angle, frequency, scale, *, shift=False):
    diagonal = 0.03 < angle % (np.pi / 2) < np.pi / 2 - 0.03
    fold = scale < 3 and (diagonal or scale > 1)
    grid = grids[768 if fold else 256]
    band = _log_kernel(grid, angle, frequency)
    if shift:
        row_direction = -np.sin(angle % np.pi)
        column_direction = np.cos(angle % np.pi)
        denominator = 256 / max(abs(row_direction), abs(column_direction))
        phase = np.pi * (grid[3] * row_direction + grid[4] * column_direction) / denominator
        band = band * np.exp(1j * phase)
    if fold:
        result = np.zeros((256, 256), dtype=band.dtype)
        position = np.arange(256) / 256
        attenuation = (1 - position, np.zeros(256), position)
        for row in range(3):
            for column in range(3):
                distance = np.hypot(attenuation[row][:, None], attenuation[column][None, :])
                weight = (np.cos(np.minimum(3 * np.pi * distance, np.pi)) + 1) / 2
                result += band[row * 256:(row + 1) * 256, column * 256:(column + 1) * 256] * weight
        band = result
    return band


class _LogGaborBank:
    """Cache the selected toolbox filters; linear_pyramid matches the old seam."""

    shape = (256, 256)
    sf_0 = np.array([0.5, 0.25, 0.125, 0.0625])  # nominal cycles/pixel
    theta = np.pi / 6 * (np.arange(6)[:, None] + np.array([0, 0.5, 0, 0.5]))

    def __init__(self):
        grids = {size: _polar_grid(size) for size in (256, 768)}
        self._filters = np.empty((*self.shape, 6, 4), dtype=np.complex128)
        for scale, frequency in enumerate(self.sf_0, start=1):
            for orientation, angle in enumerate(self.theta[:, scale - 1]):
                band = _channel(grids, angle, 256 * frequency, scale)
                if scale == 1:
                    opposite = _channel(grids, angle - np.pi, 256 * frequency, scale)
                    even = np.maximum(band, opposite) / 2
                    odd = (
                        _channel(grids, angle, 256 * frequency, scale, shift=True)
                        - _channel(grids, angle - np.pi, 256 * frequency, scale, shift=True)
                    ) / 2
                    band = even + odd
                self._filters[:, :, orientation, scale - 1] = band

        # The source builds a residual from three additional scales before
        # normalizing the whole bank; discarding it earlier changes all gains.
        residual_power = np.zeros(self.shape)
        for offset in range(3):
            for orientation in range(6):
                angle = np.pi / 6 * (orientation + (offset % 2) / 2)
                band = _channel(grids, angle, 8 / 2 ** offset, 5)
                residual_power += band ** 2 + _mirror_spectrum(band, include_edges=False) ** 2
        lowpass = np.sqrt(residual_power)
        radius = grids[256][0]
        peak = lowpass.max()
        lowpass[(radius < 8 / np.sqrt(2)) & (lowpass < peak * 0.96)] = peak
        lowpass[0, :] = 0
        lowpass[:, 0] = 0
        power = lowpass ** 2
        for scale in range(4):
            for orientation in range(6):
                band = self._filters[:, :, orientation, scale]
                power += np.abs(band) ** 2 + np.abs(_mirror_spectrum(band)) ** 2
        denominator = np.sqrt(np.maximum(power, 1e-19))
        self._filters /= denominator[:, :, None, None]
        self._lowpass = lowpass / denominator
        self._filters[128, 128, :, :] = 0

        # Reproduce the source's rcmp calibration, without a reconstruction API:
        # reconstruct a unit spatial coefficient and measure peak-to-peak gain.
        unit = np.zeros(self.shape)
        unit[127, 127] = 1
        unit_spectrum = np.fft.fftshift(np.fft.fft2(unit))
        self._amplitudes = np.empty((6, 4))
        for scale in range(4):
            for orientation in range(6):
                contribution = np.conj(self._filters[:, :, orientation, scale]) * unit_spectrum
                impulse = np.fft.ifft2(np.fft.ifftshift(contribution + _mirror_spectrum(contribution)))
                self._amplitudes[orientation, scale] = max(np.ptp(impulse.real), np.ptp(impulse.imag))
        if not (
            np.isfinite(self._filters).all() and np.isfinite(self._amplitudes).all()
            and (self._amplitudes > 0).all()
        ):
            raise FloatingPointError("Non-finite or zero-gain Log-Gabor filters were calculated")

    def linear_pyramid(self, image):
        image = _validate_image(image)
        if image.shape != self.shape:
            raise ValueError("The native Log-Gabor bank requires a 256 x 256 image; resize before filtering")
        with np.errstate(over="raise", invalid="raise"):
            spectrum = np.fft.fftshift(np.fft.fft2(image))
            responses = np.fft.ifft2(
                np.fft.ifftshift(spectrum[:, :, None, None] * self._filters, axes=(0, 1)),
                axes=(0, 1),
            ) * self._amplitudes
        if not np.isfinite(responses).all():
            raise FloatingPointError("Non-finite Log-Gabor responses were calculated")
        return responses
