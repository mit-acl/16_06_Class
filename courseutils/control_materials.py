"""
control_materials.py

Control utilities for 16.06.
All environment/setup is opt-in via setup_environment().
"""

__version__ = "16.06-0.1"

from pathlib import Path
import numpy as np
import cmath
from numpy.polynomial import Polynomial
from numpy import inf
import matplotlib.pyplot as plt

# control is an optional dependency; checked in setup_environment
import importlib.util

# constants
r2d = 180.0 / np.pi
tpi = 2 * np.pi

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

# -------------------------------
# Environment helpers
# -------------------------------

def _require_package(name):
    if importlib.util.find_spec(name) is None:
        raise ImportError(
            f"Required package '{name}' not found. "
            "Please install it following the course instructions."
        )

def setup_environment(*, verbose=False):
    """
    Opt-in environment setup for control_materials.
    - checks that control, scipy, sympy are available
    - sets control plotting defaults (if control is available)
    - configures matplotlib fonts/sizes consistent with course
    """
    # check required packages
    _require_package("control")
    _require_package("scipy")
    _require_package("sympy")

    # now import control and set defaults
    import control as ct
    # set defaults for Nyquist plotting (explicit, not on import)
    try:
        ct.set_defaults("nyquist", max_curve_magnitude=100)
    except Exception:
        # not fatal; just a best-effort setting
        pass

    # matplotlib style consistent with basic_material
    from matplotlib import rcParams
    rcParams["font.serif"] = "cmr14"
    rcParams.update({"font.size": 18})
    plt.rcParams["figure.figsize"] = [8, 5.0]
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["axes.xmargin"] = 0
    plt.rcParams["axes.grid"] = True
    plt.rcParams["figure.autolayout"] = True

    if verbose:
        import sympy as sym
        import platform
        print("control_materials: environment set")
        print("Python:", platform.python_version())
        print("SymPy:", sym.__version__)

# -------------------------------
# Utility functions
# -------------------------------

def Root_Locus_gains(L, Krange=None, Tol=1e-3, standard_locus=True, Tol_max=1e3):
    """
    Augment RL gains to include break-in/break-out points; returns augmented Krange.
    """
    if Krange is None:
        Krange = (2 * standard_locus - 1) * np.logspace(-3, 3, num=2000)
    Krange = np.sort(np.append(Krange, 0))  # add zero

    try:
        Num = L.num[0][0]
        Den = L.den[0][0]
        dNds = np.polyder(Num) if len(Num) > 1 else np.array([0.0])
        dDds = np.polyder(Den) if len(Den) > 1 else np.array([0.0])

        part1 = np.convolve(dNds, Den)
        part2 = np.convolve(Num, dDds)

        max_len = max(len(part1), len(part2))
        part1 = np.pad(part1, (max_len - len(part1), 0), "constant")
        part2 = np.pad(part2, (max_len - len(part2), 0), "constant")

        pdr = np.roots(part1 - part2)  # candidate points where dL/ds poles occur

        Kkeep = [-1.0 / np.real(L(x)) for x in pdr if abs(x.imag) < Tol]
        if standard_locus:
            Kkeep = [x for x in Kkeep if (x >= 0) and (x < Tol_max)]
        else:
            Kkeep = [x for x in Kkeep if (x <= 0) and (x > -Tol_max)]

        if len(Kkeep) > 0:
            Krange = np.sort(np.append(Krange, Kkeep))
            npoles = len(L.den[0][0])
            nzeros = len(L.num[0][0])
            n_add = int(npoles - nzeros)
            L_num_add = np.pad(L.num[0][0], (n_add, 0), "constant", constant_values=(0,))
            for kk in Kkeep:
                phi_temp = L.den[0][0] + kk * L_num_add
                scl = np.roots(phi_temp)
                real_poles = [np.round(x.real, 3) for x in scl if abs(x.imag) < Tol]
                double_real_poles = set([x for x in real_poles if real_poles.count(x) > 1])
                # Preserve informative prints only when debugging; function is quiet by default
                # print("\nFound breakin/out at K = {:4.3f}".format(kk))
                # print("At possible locations s = " + ', '.join('{:4.3f}'.format(x.real) for x in double_real_poles))
        else:
            double_real_poles = []
    except Exception:
        # fail silently and return the original Krange
        return Krange

    return Krange

def RL_COM(L, standard_locus=True):
    """
    Return center of mass and angle for root locus asymptotes.
    """
    npoles = len(L.poles())
    nzeros = len(L.zeros())

    if npoles <= nzeros:
        return None, None
    if npoles == (nzeros + 1):
        return None, 180.0 if standard_locus else 0.0

    CoM = (sum(L.poles()) - sum(L.zeros())) / (npoles - nzeros)
    Ang = (180.0 / (npoles - nzeros)) % 360.0 if standard_locus else (360.0 / (npoles - nzeros)) % 360.0
    return CoM, Ang

def Root_Locus_design_cancel(G, s_target=complex(-1, 2), s_cancel=-1, verbose=False):
    """
    Root locus lead design by cancelling/placing pole at s_cancel to get CL poles at s_target.
    Returns (Gc, Gcl_poles).
    """
    import control as ct
    from control.matlab import tf, feedback

    phi_fromG = sum([cmath.phase(x) for x in (s_target - G.zeros())]) * r2d - \
                sum([cmath.phase(x) for x in (s_target - G.poles())]) * r2d

    Gczeros = np.array([np.real(s_cancel)])
    phi_from_Gc_zero = sum([cmath.phase(x) for x in (s_target - Gczeros)]) * r2d
    phi_required = (180 + phi_fromG + phi_from_Gc_zero) % 360

    P = s_target.imag / np.tan(phi_required / r2d) - s_target.real
    Gc = tf((1, -Gczeros[0]), (1, P))
    Gain = -1.0 / np.real(G(s_target) * Gc(s_target))
    Gc *= Gain
    L = G * Gc
    Gcl = feedback(L, 1)

    return Gc, Gcl.poles()

def Root_Locus_design_ratio(G, s_target=complex(-1, 2), gamma=10, z0=None, idx=None, verbose=False):
    """
    Root locus design using zero/pole ratio gamma.
    Returns (Gc, Gcl_poles).
    """
    import control as ct
    from control.matlab import tf, feedback
    from scipy.optimize import minimize

    def func(z, gam, G, s_0):
        Gc = tf((1, float(z)), (1, float(gam * z)))
        L = Gc * G
        phi_fromL = (sum([cmath.phase(x) for x in (s_0 - L.zeros())]) * r2d -
                     sum([cmath.phase(x) for x in (s_0 - L.poles())]) * r2d) % 360
        return (phi_fromL - 180) % 360

    if z0 is None:
        z0 = -s_target.real / 2

    res = minimize(func, x0=z0, args=(gamma, G, s_target), tol=1e-3, method="Nelder-Mead",
                   options={"disp": verbose, "maxiter": 1000})
    if not res.success:
        raise RuntimeError("Optimization failed")
    Gczeros = res.x[idx] if idx is not None else res.x[0]
    Gc = tf((1, float(Gczeros)), (1, float(gamma * Gczeros)))
    Gain = -1.0 / np.real(G(s_target) * Gc(s_target))
    Gc *= Gain
    L = G * Gc
    Gcl = feedback(L, 1)
    return Gc, Gcl.poles()

def Root_Locus_design_PD(G, s_target=complex(-1, 2), verbose=False):
    """
    PD design to place CL poles at s_target.
    Returns (Gc, Gcl_poles).
    """
    from control.matlab import tf, feedback

    phi_fromG = sum([cmath.phase(x) for x in (s_target - G.zeros())]) - \
                sum([cmath.phase(x) for x in (s_target - G.poles())])
    phi_required = (np.pi - phi_fromG) % (2 * np.pi)

    Z = s_target.imag / np.tan(phi_required) - s_target.real
    Gc = tf((1, Z), 1)
    Gain = -1.0 / np.real(G(s_target) * Gc(s_target))
    Gc *= Gain
    L = G * Gc
    Gcl = feedback(L, 1)
    return Gc, Gcl.poles()

# -------------------------------
# Step info class (keeps API but safer)
# -------------------------------

class Step_info:
    def __init__(self, t, y, method=0, t0=0, SettlingTimeLimits=None, RiseTimeLimits=(0.1, 0.9)):
        self.t = np.asarray(t)
        self.y = np.asarray(y)
        self.Yss = self.y[-1]
        if SettlingTimeLimits is None:
            SettlingTimeLimits = [0.02]
        self.SettlingTimeLimits = SettlingTimeLimits
        self.RiseTimeLimits = RiseTimeLimits
        sgnYss = np.sign(self.Yss.real) if np.isreal(self.Yss) else np.sign(self.Yss)

        tr_lower_index = np.where(sgnYss * (self.y - RiseTimeLimits[0] * self.Yss) >= 0)[0][0]
        tr_upper_index = np.where(sgnYss * (self.y - RiseTimeLimits[1] * self.Yss) >= 0)[0][0]
        self.Tr = self.t[tr_upper_index] - self.t[tr_lower_index]
        self.Tr_values = [self.t[tr_lower_index] - t0, self.t[tr_upper_index] - t0]

        settled = np.where(np.abs(self.y / self.Yss - 1) >= SettlingTimeLimits[0])[0]
        self.Ts = (self.t[settled[-1]] - t0) if settled.size > 0 and (settled[-1] + 1) < len(self.t) else 0.0

        self.Mp = (np.max(self.y) / self.Yss - 1)
        max_idx = int(np.argmax(self.y))
        self.Tp = float(self.t[max_idx]) - t0

        if method == 0:
            # using Tp
            if self.Mp <= 0:
                self.zeta = np.nan
                self.wn = np.nan
            else:
                self.zeta = 1.0 / np.sqrt(1.0 + (np.pi / np.log(self.Mp)) ** 2)
                self.wn = np.pi / self.Tp / np.sqrt(1.0 - self.zeta ** 2)
        else:
            q = self.Tp / np.pi / self.Ts if self.Ts != 0 else np.nan
            if self.SettlingTimeLimits[0] == 0.01:
                q *= 4.6
            else:
                q *= 4.0
            self.zeta = q / np.sqrt(1.0 + q ** 2) if not np.isnan(q) else np.nan
            self.wn = 4.0 / self.Ts / self.zeta if self.zeta != 0 else np.nan

    def printout(self, raw=False):
        print(f"omega_n:\t{self.wn:.3f}")
        print(f"zeta   :\t{self.zeta:.3f}")
        print(f"Tr     :\t{self.Tr:.2f}s")
        print(f"Ts     :\t{self.Ts:.2f}s")
        print(f"Mp     :\t{self.Mp:.2f}")
        print(f"Tp     :\t{self.Tp:.2f}s")
        print(f"Yss    :\t{self.Yss:.2f}")

    def nice_plot(self, ax=None, Tmax=None, Ymax=None):
        if Ymax is None:
            ylim = (np.floor(np.min(self.y)), np.ceil(10.0 * np.max(self.y)) / 10.0)
            Ymax = np.max(ylim)
        if Tmax is None:
            Tmax = np.max(self.t)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(self.t, self.y, "b")
        ax.axvline(x=self.Tr_values[0], ymax=0.1 * self.Yss / Ymax, c="r", ls="dashed")
        ax.axvline(x=self.Tr_values[1], ymax=0.9 * self.Yss / Ymax, c="r", ls="dashed")
        ax.axvline(x=self.Ts, ymax=self.Yss / Ymax, c="grey", ls="dashed")
        ax.axvline(ymax=self.Yss * (1 + self.Mp) / Ymax, x=self.Tp, c="m", ls="dashed", lw=2)
        ax.axhline(y=(1 + self.SettlingTimeLimits[0]) * self.Yss, xmin=self.Ts / Tmax, c="grey", ls="dashed", lw=1)
        ax.axhline(y=(1 - self.SettlingTimeLimits[0]) * self.Yss, xmin=self.Ts / Tmax, c="grey", ls="dashed", lw=1)
        ax.plot((0, self.Tp), (self.Yss * (1 + self.Mp), self.Yss * (1 + self.Mp)), c="green", ls="dashed", lw=2)
        ax.text(self.Tr / 2, 0.25 * self.Yss, f"Tr = {self.Tr:.2f}", fontsize=SMALL_SIZE)
        ax.text(self.Tp, 0.75 * self.Yss, f"Tp = {self.Tp:.2f}", fontsize=SMALL_SIZE)
        ax.text(self.Ts, 0.5 * self.Yss, f"Ts = {self.Ts:.2f}", fontsize=SMALL_SIZE)
        ax.text(self.Tp * 1.1, self.Yss * (1 + self.Mp), f"Mp = {self.Mp:.2f}", fontsize=SMALL_SIZE)
        ax.text(Tmax * 0.6, self.Yss * 0.9, rf"$e_{{ss}}$ = {1 - self.Yss:.3f}", fontsize=SMALL_SIZE, color="purple")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("Response")
        ax.set_title("Step Response")
        ax.set_ylim(0, Ymax)
        ax.set_xlim(0, Tmax)

# -------------------------------
# More utilities
# -------------------------------

def lead_design(G, wc_des=1, PM=45, verbose=False):
    import cmath
    from control.matlab import tf

    j = complex(0, 1)
    Gf = G(j * wc_des)
    phi_G = cmath.phase(Gf) * r2d
    if phi_G > 0:
        phi_G -= 360
    phi_m = (PM - (180 + phi_G)) / r2d

    zdp = (1.0 - np.sin(phi_m)) / (1.0 + np.sin(phi_m))
    z = float(wc_des * np.sqrt(zdp))
    p = float(z / zdp)

    Gc_lead = tf([1, z], [1, p])
    L = G * Gc_lead
    k_c = 1.0 / np.abs(L(j * wc_des))
    Gc_lead *= k_c

    latex_paragraph = (
        f"The phase of the open-loop transfer function $G(j\\omega_c)$ at the desired crossover frequency "
        f"is $\\phi_G = {phi_G:.2f}^\\circ$. Thus the required phase lead is calculated as "
        f"$\\phi_m = {phi_m * r2d:.2f}^\\circ$. Using the phase lead equation $$\\dfrac{{z}}{{p}} = "
        f"\\dfrac{{1 - \\sin(\\phi_m)}}{{1 + \\sin(\\phi_m)}} = {zdp:.3f}.$$ The zero and pole of the lead "
        f"compensator are then placed at $z = {z:.2f}$ and $p = {p:.2f}$, respectively. Finally, the "
        f"compensator gain is adjusted to achieve the desired crossover frequency, resulting in "
        f"$k_c = {k_c:.2f}$. The resulting lead compensator transfer function is "
        f"$G^{{lead}}_c(s) = {k_c:.2f}\\dfrac{{s+{z:.2f}}}{{s+{p:.2f}}}$"
    )

    if verbose:
        return Gc_lead, latex_paragraph
    else:
        return Gc_lead

def lag_design(gain_inc=10, gamma=10, wc=1, verbose=False):
    from control.matlab import tf

    zl = float(wc / gamma)
    pl = float(zl / gain_inc)
    if verbose:
        latex_paragraph = (
            f"The lag compensator zero at $z_l = {zl:.2f}$ and pole at $p_l = {pl:.3f}$. "
            f"Resulting lag compensator $G^{{lag}}_c(s) = \\dfrac{{s+{zl:.2f}}}{{s+{pl:.3f}}}$"
        )
        return tf([1, zl], [1, pl]), latex_paragraph
    else:
        return tf([1, zl], [1, pl])

def find_system_type(L):
    return len(L.den[0][0]) - len(np.trim_zeros(L.den[0][0], "b"))

def find_Kp(L):
    L_type = find_system_type(L)
    if L_type == 0:
        return float(np.real(L.num[0][0][-1] / L.den[0][0][-1]))
    return None

def find_Kv(L):
    L_type = find_system_type(L)
    if L_type == 0:
        return 0.0
    if L_type == 1:
        return float(np.real(L.num[0][0][-1] / L.den[0][0][-2]))
    return None

def find_Ka(L):
    L_type = find_system_type(L)
    if L_type < 2:
        return 0.0
    if L_type == 2:
        return float(np.real(L.num[0][0][-1] / L.den[0][0][-3]))
    return None

def find_wc(omega, G, mag=1.0):
    Gf = G(1j * omega)
    idx = np.argmin(np.abs(mag - np.abs(Gf)))
    return omega[idx], idx

def find_wpi(omega, G, phi=180, verbose=False):
    Gf = G(1j * omega)
    idx = np.argmin(np.abs(phi - (np.angle(Gf) * r2d)))
    if verbose:
        print(f"wpi = {omega[idx]:.3f} r/s idx = {idx}")
        print(f"ang G(jwpi) = {np.angle(G(1j * omega[idx])) * r2d:.3f}")
    return omega[idx], idx

def pshift(Gp):
    Gp = np.asarray(Gp)
    while (np.max(Gp) < -np.pi):
        Gp += 2 * np.pi
    while (np.min(Gp) > np.pi):
        Gp -= 2 * np.pi
    return Gp

def caption(txt, fig, xloc=0.5, yloc=-0.05):
    fig.text(xloc, yloc, txt, ha="center", size=MEDIUM_SIZE, color="blue")

def my_pzmap(G, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.real(G.poles()), np.imag(G.poles()), "bx", ms=6)
    ax.plot(np.real(G.zeros()), np.imag(G.zeros()), "o", ms=6, markeredgewidth=2,
            markeredgecolor="r", markerfacecolor="r")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Pole-Zero Map")
    ax.grid(True)
    return ax

def color_rl(ax):
    for line in ax.lines:
        if line.get_linestyle() == "-":
            line.set_linewidth(1.5)
            line.set_color("blue")
        if line.get_marker() == "x":
            line.set_markersize(8)
            line.set_color("blue")
        if line.get_marker() == "o":
            line.set_markersize(8)
            line.set_markerfacecolor("r")
            line.set_markeredgecolor("r")
        if line.get_marker() == "d":
            line.set_markersize(8)
            line.set_markerfacecolor("g")
            line.set_markeredgecolor("g")

def Read_data(file_name, comments=["#", "F"], cols=[0]):
    return np.loadtxt(file_name, comments=comments, delimiter=",", usecols=cols)

def near_zero(P, Tol=1e-12):
    P.num[0][0] = [x if abs(x) > Tol else 0.0 for x in P.num[0][0]]
    P.den[0][0] = [x if abs(x) > Tol else 0.0 for x in P.den[0][0]]
    import control
    return control.tf(P.num, P.den)

def log_interp(zz, xx, yy):
    logz = np.log10(zz)
    logx = np.log10(xx)
    logy = np.log10(yy)
    return np.power(10.0, np.interp(logz, logx, logy))

# balanced truncation
from scipy.linalg import solve_continuous_lyapunov, svd
def balred(G, order=None, DCmatch=False, check=False, Tol=1e-5):
    import control
    # convert to TF if state-space provided
    if not isinstance(G, control.StateSpace):
        convert_to_TF = True
        Gin = G
    else:
        convert_to_TF = False
        Gin = control.ss2tf(G)

    G_trimmed = control.tf(Gin.num[0][0], np.trim_zeros(Gin.den[0][0], "b"))
    number_cut = len(Gin.den[0][0]) - len(G_trimmed.den[0][0])

    Gss = control.tf2ss(G_trimmed)
    if order is None:
        order = Gss.A.shape[0] - 1
    order -= number_cut

    Wc = solve_continuous_lyapunov(Gss.A, -Gss.B @ Gss.B.T)
    Wo = solve_continuous_lyapunov(Gss.A.T, -Gss.C.T @ Gss.C)

    U = np.linalg.cholesky(Wc)
    Z = np.linalg.cholesky(Wo)
    W, S, Vh = svd(U.T @ Z)
    S_sqrt_inv = np.linalg.inv(np.diag((np.sqrt(S))))

    T = S_sqrt_inv @ Vh @ Z.T
    Ti = U @ W @ S_sqrt_inv

    Ab = T @ Gss.A @ Ti
    Bb = T @ Gss.B
    Cb = Gss.C @ Ti

    Arr = Ab[:order, :order]
    Brr = Bb[:order, :]
    Crr = Cb[:, :order]
    Drr = Gss.D
    Gr = None
    try:
        Gr = control.matlab.StateSpace(Arr, Brr, Crr, Drr)
    except Exception:
        Gr = control.StateSpace(Arr, Brr, Crr, Drr)

    if DCmatch:
        try:
            Are = Ab[:order, order:]
            Aer = Ab[order:, :order]
            Aee = Ab[order:, order:]
            Be = Bb[order:, :]
            Ce = Cb[:, order:]
            Aee_inv = np.linalg.inv(Aee)
            Arr -= Are @ Aee_inv @ Aer
            Brr -= Are @ Aee_inv @ Be
            Crr -= Ce @ Aee_inv @ Aer
            Drr -= Ce @ Aee_inv @ Be
            Gr = control.matlab.StateSpace(Arr, Brr, Crr, Drr)
        except Exception:
            pass

    Gr = near_zero(control.ss2tf(Gr)) * control.tf([1], [1, 0]) ** number_cut
    return Gr if convert_to_TF else control.tf2ss(Gr)

def pretty_row_print(X, msg=""):
    print(msg + ", ".join("({0.real:.2f} + {0.imag:.2f}i)".format(x) if np.iscomplex(x) else "{:.3f}".format(x.real) for x in X))

def feedback_ff(G, K, Kff):
    import control
    from control.matlab import tf
    if not isinstance(G, control.TransferFunction):
        G = control.tf(G)
    if not isinstance(K, control.TransferFunction):
        K = control.tf(K)

    NG = G.num[0][0]
    DG = G.den[0][0]
    NC = K.num[0][0]
    DC = K.den[0][0]

    NGDC = np.convolve(NG, DC)
    NGNC = np.convolve(NG, NC)
    DGDC = np.convolve(DG, DC)

    max_len = max(len(DGDC), len(NGNC), len(NGDC))
    NGNC = np.pad(NGNC, (max_len - len(NGNC), 0), "constant")
    NGDC = np.pad(NGDC, (max_len - len(NGDC), 0), "constant")
    DGDC = np.pad(DGDC, (max_len - len(DGDC), 0), "constant")

    return tf(Kff * NGDC + NGNC, DGDC + NGNC)

def writeGc(filename, Gc):
    """
    Write controller info to filename. Each piece on its own line:
     - zeros (real parts comma separated)
     - poles (real parts comma separated)
     - DC gain (single number)
    """
    zs = [float(np.real(z)) for z in Gc.zeros()]
    ps = [float(np.real(p)) for p in Gc.poles()]
    gain = float(Gc.num[0][0][0] / Gc.den[0][0][0]) if (len(Gc.num[0][0]) and len(Gc.den[0][0])) else 0.0

    with open(filename, "w") as f:
        f.write("zeros:" + ",".join(f"{z:4.2f}" for z in zs) + "\n")
        f.write("poles:" + ",".join(f"{p:4.2f}" for p in ps) + "\n")
        f.write("gain:" + f"{gain:4.2f}" + "\n")

# module is quiet on import