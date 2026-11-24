"""2D incompressible flow on a periodic box [0, 2pi)^2, vorticity form, pseudo-spectral (2/3 dealiasing), RK4.

    d(omega)/dt = -u.grad(omega) + nu*lap(omega) - mu*omega + F*sin(kf*y) - beta*v

with streamfunction psi (lap psi = -omega), u = d(psi)/dy, v = -d(psi)/dx. The parameters are the hidden "law" that a
video model trained on frames of omega must learn, and that Kepler must read back out of the model.

Fields are arrays [..., Ny, Nx]: axis -2 is y (rows), axis -1 is x (columns).
"""
import json
import numpy as np


class Spectral:
    """Wavenumbers and exact spectral operators for an N x N periodic grid on [0, 2pi)^2."""

    def __init__(self, n=64):
        self.n = n
        kx = np.fft.rfftfreq(n, d=1.0 / n)            # 0..n/2
        ky = np.fft.fftfreq(n, d=1.0 / n)             # 0..n/2-1, -n/2..-1
        self.KX, self.KY = np.meshgrid(kx, ky)        # [n, n//2+1]
        self.K2 = self.KX ** 2 + self.KY ** 2
        self.K2i = np.where(self.K2 > 0, 1.0 / np.where(self.K2 > 0, self.K2, 1.0), 0.0)
        self.mask = ((np.abs(self.KX) < n / 3.0) & (np.abs(self.KY) < n / 3.0)).astype(float)
        x = np.arange(n) * (2 * np.pi / n)
        self.X, self.Y = np.meshgrid(x, x)            # X varies along columns, Y along rows

    def fft(self, f):
        return np.fft.rfft2(f)

    def ifft(self, fh):
        return np.fft.irfft2(fh, s=(self.n, self.n))

    # --- exact derived fields from omega (physical space in, physical space out) ---
    def fields(self, w):
        """Every physical quantity the concept library and the law library use, computed exactly from omega."""
        wh = self.fft(w) * self.mask
        psih = wh * self.K2i
        I = 1j
        out = {
            "omega": self.ifft(wh),
            "psi": self.ifft(psih),
            "u": self.ifft(I * self.KY * psih),
            "v": self.ifft(-I * self.KX * psih),
            "omega_x": self.ifft(I * self.KX * wh),
            "omega_y": self.ifft(I * self.KY * wh),
            "lap_omega": self.ifft(-self.K2 * wh),
            "bilap_omega": self.ifft(self.K2 ** 2 * wh),
        }
        ux = self.ifft(I * self.KX * I * self.KY * psih)          # du/dx = psi_xy
        uy = self.ifft(I * self.KY * I * self.KY * psih)          # du/dy = psi_yy
        vx = self.ifft(-I * self.KX * I * self.KX * psih)         # dv/dx = -psi_xx
        out["adv"] = out["u"] * out["omega_x"] + out["v"] * out["omega_y"]   # u.grad(omega)
        out["strain_n"] = ux                                       # normal strain (du/dx = -dv/dy)
        out["strain_s"] = 0.5 * (uy + vx)                          # shear strain
        out["okubo_weiss"] = 4 * (ux ** 2 + out["strain_s"] ** 2) - out["omega"] ** 2
        out["ke"] = 0.5 * (out["u"] ** 2 + out["v"] ** 2)
        out["grad_omega_sq"] = out["omega_x"] ** 2 + out["omega_y"] ** 2
        return out


class Flow:
    def __init__(self, n=64, nu=0.01, mu=0.05, F=0.5, kf=4, beta=0.6, dt=0.0125):
        self.sp = Spectral(n)
        self.nu, self.mu, self.F, self.kf, self.beta, self.dt = nu, mu, F, kf, beta, dt
        self.fh = self.sp.fft(F * np.sin(kf * self.sp.Y)) * self.sp.mask

    def params(self):
        return {"nu": self.nu, "mu": self.mu, "F": self.F, "kf": self.kf, "beta": self.beta, "dt": self.dt,
                "n": self.sp.n}

    def rhs_hat(self, wh):
        sp = self.sp
        psih = wh * sp.K2i
        uh, vh = 1j * sp.KY * psih, -1j * sp.KX * psih
        u, v = sp.ifft(uh), sp.ifft(vh)
        wx, wy = sp.ifft(1j * sp.KX * wh), sp.ifft(1j * sp.KY * wh)
        advh = sp.fft(u * wx + v * wy) * sp.mask
        return -advh - (self.nu * sp.K2 + self.mu) * wh + self.fh - self.beta * vh

    def rhs(self, w):
        """Exact time derivative in physical space (the ground truth the law extractor is tested on)."""
        return self.sp.ifft(self.rhs_hat(self.sp.fft(w) * self.sp.mask))

    def step_hat(self, wh, nsub):
        dt = self.dt
        for _ in range(nsub):
            k1 = self.rhs_hat(wh)
            k2 = self.rhs_hat(wh + 0.5 * dt * k1)
            k3 = self.rhs_hat(wh + 0.5 * dt * k2)
            k4 = self.rhs_hat(wh + dt * k3)
            wh = (wh + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)) * self.sp.mask
        return wh

    def random_init(self, rng, batch, amp_range=(0.5, 4.0), k_peak_range=(2.0, 8.0)):
        """Random vorticity fields with a peaked spectrum and random rms amplitude (variety of regimes)."""
        sp = self.sp
        k = np.sqrt(sp.K2)
        out = np.empty((batch, sp.n, sp.n))
        for b in range(batch):
            kp = rng.uniform(*k_peak_range)
            spec = (k / kp) ** 4 * np.exp(-2.0 * (k / kp) ** 2)
            ph = rng.normal(size=sp.K2.shape) + 1j * rng.normal(size=sp.K2.shape)
            wh = np.sqrt(spec) * ph * sp.mask
            wh[0, 0] = 0
            w = sp.ifft(wh)
            w *= rng.uniform(*amp_range) / (w.std() + 1e-12)
            out[b] = w
        return out

    def trajectories(self, rng, batch, frames, frame_dt, spinup=0.0):
        """Return omega frames [batch, frames, n, n] sampled every frame_dt (float32)."""
        nsub = int(round(frame_dt / self.dt))
        assert abs(nsub * self.dt - frame_dt) < 1e-9, "frame_dt must be a multiple of dt"
        wh = self.sp.fft(self.random_init(rng, batch)) * self.sp.mask
        if spinup > 0:
            wh = self.step_hat(wh, int(round(spinup / self.dt)))
        out = np.empty((batch, frames, self.sp.n, self.sp.n), dtype=np.float32)
        for t in range(frames):
            out[:, t] = self.sp.ifft(wh)
            if t < frames - 1:
                wh = self.step_hat(wh, nsub)
        return out


def seal(params, path):
    """Write the true parameters. Only tests open this file; the engine never reads it."""
    with open(path, "w") as f:
        json.dump(params, f, indent=1)
