"""
control_materials.py

Control utilities for 16.06.
All environment/setup is opt-in via setup_environment().
"""

__version__ = "16.06-0.5"

import numpy as np
import cmath

import matplotlib.pyplot as plt
import sympy as sp
import control as ct
import control.matlab as cmat

import importlib.util
from dataclasses import dataclass
from typing import List
from IPython.display import Math
import scipy.linalg
import re

from types import SimpleNamespace

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

    # stop annoying warnings
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="divide by zero encountered in divide"
    )
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in divide"
    )
    import logging
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    # matplotlib style consistent with basic_material
    from matplotlib import rcParams
    rcParams["font.serif"] = "cmr14"
    rcParams.update({"font.size": 10})
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

def Root_Locus_gains(L, Krange=None, Tol=1e-5, standard_locus=True, Tol_max=1e3, verbose = False, debug=None):
    """
    Augment RL gains to include break-in/break-out points; returns augmented Krange.

    Tol_max - limits how large the figure will be
    verbose - enables additional return of break info
    debug - extensive on screen information
    """
    # Basic checks
    if not isinstance(L, ct.TransferFunction):
        raise TypeError("Root_Locus_gains expects a control.TransferFunction (SISO).")

    if Krange is None:
        # provide positive or negative range of K values
        Krange = (2 * standard_locus - 1) * np.logspace(-3, 3, num=2000)
    Krange = np.sort(np.append(Krange, 0))  # add zero

    break_info = [] 
    
    @dataclass
    class BreakPoint:
        K: float
        poles: List[float]

    try:
        Num = np.asarray(L.num[0][0], dtype=float)
        Den = np.asarray(L.den[0][0], dtype=float)
        npoles = len(L.den[0][0])
        nzeros = len(L.num[0][0])
        n_add = int(npoles - nzeros)
        L_num_add = np.pad(L.num[0][0], (n_add, 0), "constant", constant_values=(0,))

        dNds = np.polyder(Num) if len(Num) > 1 else np.array([0.0])
        dDds = np.polyder(Den) if len(Den) > 1 else np.array([0.0])

        part1 = np.convolve(dNds, Den)
        part2 = np.convolve(Num, dDds)

        max_len = max(len(part1), len(part2))
        part1 = np.pad(part1, (max_len - len(part1), 0), "constant")
        part2 = np.pad(part2, (max_len - len(part2), 0), "constant")

        pdr = np.roots(part1 - part2)  # candidate points where dL/ds poles occur
        if debug:
            pretty_row_print(pdr,"All possible locations: ")

        Kkeep = [-1.0 / np.real(L(x)) for x in pdr if (abs(x.imag) < Tol and np.abs(np.real(L(x))) > Tol)]
        if standard_locus:
            Kkeep = [x for x in Kkeep if (x >= 0) and (x < Tol_max)]
        else:
            Kkeep = [x for x in Kkeep if (x <= 0) and (x > -Tol_max)]

        if debug:
            pretty_row_print(Kkeep, "Associated gains ")        

        def is_real_pole(p, atol=Tol * 1e-3, rtol=Tol):
            return abs(p.imag) <= atol + rtol * abs(p.real)

        # find the location of the break in/out pts -- given by duplicate real poles
        if len(Kkeep) > 0:
            for kk in np.array(Kkeep):
                phi_temp = L.den[0][0] + kk * L_num_add

                scl = np.roots(phi_temp) # clp poles for that gain
                if debug:
                    print(f"For gain {kk:5.3f} poles at ",scl)
                    print("Are the poles real? ", is_real_pole(scl,rtol=Tol))

                real_poles = np.real(scl[is_real_pole(scl)])
                if debug:
                    pretty_row_print(real_poles, "Real poles found")

                # which real are double?
                double_real_poles = find_double_real_poles(real_poles, tol=Tol)

                if len(double_real_poles) > 0:
                    if debug:
                        print("Double Real gain, pole location? ",kk,double_real_poles,np.diff(real_poles))
                    break_info.append(
                        BreakPoint(
                            K = float(kk),
                            poles = [float(p) for p in double_real_poles]
                        )
                    )
                    if not verbose:
                        print(f"\nFound break-in/out at K = {kk:6.3f}")
                        print("At possible locations s = "
                            + ", ".join(f"{p:6.3f}" for p in double_real_poles))

                    Krange = np.sort(np.append(Krange, kk)) # add gains associated with double real poles

                else:
                    if debug:
                        print("Not Double? {kk:5.2f}",real_poles,double_real_poles,np.diff(real_poles))
                    else:
                        pass


    except Exception as e:
        print("failed to find Krange:", e)

    Krange = [float(k) for k in Krange]

    if verbose:
        return Krange, break_info
    else:
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

def pshift(Gp, period = 2*np.pi):
    '''shift phase to within +/-180 or +/-pi'''
    Gp = np.asarray(Gp, dtype=float)

    # detect units
    if np.max(np.abs(Gp)) > 2*np.pi + 1e-3:
        # degrees
        period = 360.0

    while np.max(Gp) < -period//2:
        Gp += period
    while np.min(Gp) > period//2:
        Gp -= period

    return Gp

def wrap(phi, period = 2*np.pi):
    '''wrap  the phase - units detected'''
    # detect units
    phi = np.asarray(phi, dtype=float)

    if np.max(np.abs(phi)) > 2*np.pi + 1e-3: # likely in degrees
        period = 360.0

    return (phi + period//2) % (period) - period//2

def wrap_phase_neg(phi, period=2*np.pi):
    """
    Wrap phase to (-period, 0].
    Works for scalars and arrays.
    Units (rad/deg) are auto-detected.
    """
    phi = np.asarray(phi, dtype=float)

    # detect units
    if np.nanmax(np.abs(phi)) > 2*np.pi + 1e-3:
        period = 360.0

    # wrap to (-period, 0]
    return (phi % period) - period

def Root_Locus_design_cancel(G, s_target=complex(-1, 2), s_cancel=-1, verbose=True):
    """
    Root locus lead design by cancelling/placing pole at s_cancel to get CL poles at s_target.
    Returns Gc, Gcl_poles()
    """

    phi_fromG = phase_at_freq(G,s_target)

    Gczeros = np.array([np.real(s_cancel)])
    phi_from_Gc_zero = sum([cmath.phase(x) for x in (s_target - Gczeros)]) * r2d
    phi_required = (180 + phi_fromG + phi_from_Gc_zero) % 360

    if verbose:
        print(f"Phase from G {phi_fromG:4.2f}")
        print(f"Phase from Gc zero {phi_from_Gc_zero:4.2f}")
        print(f"Phase required {phi_required:4.2f}")

    P = s_target.imag / np.tan(phi_required / r2d) - s_target.real
    Gc = ct.tf((1, -Gczeros[0]), (1, P))
    Gain = 1.0 / np.abs(G(s_target) * Gc(s_target))
    Gc *= Gain
    Gcl = ct.feedback(G * Gc)

    if verbose:
        return Gc, Gcl.poles(), SimpleNamespace(**{
        "phi_from_G": phi_fromG,
        "phi_from_Gc": phi_from_Gc_zero,
        "phi_required": phi_required,
    })
    else:
        return Gc, Gcl.poles()

def phase_at_freq(G,s0,modulation = None):
    '''For given G(s) and complex frequency s0, return phase of G(s0) in degrees'''
    phi_fromG = np.rad2deg(
        np.sum(np.angle(s0 - G.zeros())) -
        np.sum(np.angle(s0 - G.poles()))
    )

    if modulation == False:
        return phi_fromG 
    else:
        return phi_fromG % 360

def Root_Locus_design_ratio(G, s_target=complex(-1, 2), gamma=10, z0=None, idx=None, verbose=False):
    """
    Root locus design using zero/pole ratio gamma.
    Returns Gc, Gcl_poles()
    """
    from scipy.optimize import minimize

    def func(theta, gam, G, s_0):
        z = np.exp(theta)
        Gc = ct.tf((1, float(z)), (1, float(gam * z)))
        L = Gc * G
        phi_fromL = phase_at_freq(L,s_0)
        phase_err = wrap(phi_fromL - 180, period = 360)
        return phase_err**2

    if z0 is None:
        z0 = -s_target.real / 2

    res = minimize(func, x0=np.log(z0), args=(gamma, G, s_target), tol=1e-3, method="Nelder-Mead",
                   options={"disp": verbose, "maxiter": 1000})
    if not res.success:
        raise RuntimeError("Optimization failed")
    Gczeros = np.exp(res.x[0])

    # without gain
    Gc = ct.tf((1, float(Gczeros)), (1, float(gamma * Gczeros)))
    Gain = 1.0 / np.abs(G(s_target) * Gc(s_target))

    # apply gain
    Gc *= Gain
    L = G * Gc

    #close loop
    Gcl = ct.feedback(L)

    return Gc, Gcl.poles()

def Root_Locus_design_PD(G, s_target=complex(-1, 2), verbose=False):
    """
    PD design to place CL poles at s_target.
    Returns (Gc, Gcl_poles).
    """
    phi_fromG = phase_at_freq(g,s_target)
    phi_required = (np.pi - phi_fromG) % (2 * np.pi)

    Z = s_target.imag / np.tan(phi_required) - s_target.real

    # without gain
    Gc = ct.tf((1, Z), 1)
    Gain = -1.0 / np.real(G(s_target) * Gc(s_target))

    # apply gain
    Gc *= Gain
    L = G * Gc
    Gcl = ct.feedback(L)

    if verbose:
        return Gc, Gcl.poles(), SimpleNamespace(**{
        "phi_from_G": phi_fromG,
        "phi_required": phi_required,
        })
    else:
        return Gc, Gcl.poles()



# -------------------------------
# Step info class (keeps API but safer)
# -------------------------------

def max_overshoot(t, y, yss=None):
    """
    Compute maximum overshoot Mp and peak time Tp from step response.

    Returns
    -------
    Mp : float
        Maximum overshoot as a fraction (e.g. 0.15 for 15%)
    Tp : float
        Peak time (time at maximum overshoot)
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if yss is None:
        yss = y[-1]

    if yss == 0:
        return np.nan, np.nan

    # work in the direction of the step
    sgn = np.sign(yss)
    y_adj = sgn * y
    yss_adj = abs(yss)

    # peak relative to steady state
    idx_peak = np.argmax(y_adj)
    ymax = y_adj[idx_peak]

    Mp = (ymax - yss_adj) / yss_adj
    Mp = max(0.0, Mp)   # clip if no overshoot

    Tp = t[idx_peak] if Mp > 0 else np.nan

    return Mp, Tp

def settling_time(t, y, tol=0.02, t0=0):
    """
    Compute 2% settling time.
    Returns np.nan if never settles.
    """
    y = np.asarray(y)
    t = np.asarray(t)
    yss = y[-1]
    if yss == 0:
        return np.nan
    band = tol * abs(yss)
    err = np.abs(y - yss)
    outside = np.where(err > band)[0]     # indices where response is OUTSIDE the band
    if len(outside) == 0:
        return t[0]   # already settled
    last_outside = outside[-1]
    if last_outside == len(t) - 1:
        return np.nan  # never settles within simulation time
    return t[last_outside + 1]

import numpy as np

def rise_time(t, y, yss=None, limits=(0.1, 0.9), t0=0.0):
    """
    Robust rise time computation using linear interpolation.

    Parameters
    ----------
    t : array_like
        Time vector
    y : array_like
        Response vector
    yss : float or None
        Steady-state value (default: y[-1])
    limits : tuple
        Fractional rise limits, e.g. (0.1, 0.9)
    t0 : float
        Time offset to subtract (default: 0)

    Returns
    -------
    Tr : float
        Rise time (NaN if undefined)
    (t_lo, t_hi) : tuple
        Times at lower and upper crossings
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if yss is None:
        yss = y[-1]

    if yss == 0:
        return np.nan, (np.nan, np.nan)

    sgn = np.sign(yss)
    y_adj = sgn * y
    yss_adj = abs(yss)

    y_lo = limits[0] * yss_adj
    y_hi = limits[1] * yss_adj

    # Find first crossing indices
    def crossing_time(level):
        idx = np.where(y_adj >= level)[0]
        if len(idx) == 0 or idx[0] == 0:
            return np.nan
        i = idx[0]
        # linear interpolation
        t1, t2 = t[i-1], t[i]
        y1, y2 = y_adj[i-1], y_adj[i]
        return t1 + (level - y1) * (t2 - t1) / (y2 - y1)

    t_lo = crossing_time(y_lo)
    t_hi = crossing_time(y_hi)

    if np.isnan(t_lo) or np.isnan(t_hi):
        return np.nan, (t_lo, t_hi)

    Tr = t_hi - t_lo
    return Tr, (t_lo - t0, t_hi - t0)

class Step_info:
    def __init__(self, t, y, method=0, t0=0, SettlingTimeLimits=None, RiseTimeLimits=(0.1, 0.9)):
        self.t = np.asarray(t)
        self.y = np.asarray(y)
        self.Yss = self.y[-1] # assumes Tf is large enough that this is true

        if SettlingTimeLimits is None:
            self.SettlingTimeLimits = [0.02]
        elif np.isscalar(SettlingTimeLimits):
            self.SettlingTimeLimits = [SettlingTimeLimits]
        else:
            self.SettlingTimeLimits = list(SettlingTimeLimits)        

        self.RiseTimeLimits = RiseTimeLimits
        sgnYss = np.sign(self.Yss.real) if np.isreal(self.Yss) else np.sign(self.Yss)

        self.Tr, self.Tr_values = rise_time(self.t, self.y, yss=self.Yss, limits=RiseTimeLimits, t0=t0)
        self.Ts = settling_time(self.t, self.y, tol=self.SettlingTimeLimits[0], t0=t0) 
        self.Mp, self.Tp = max_overshoot(self.t, self.y, self.Yss)

        # different assumptions can be used here to estimate these response parameters
        if method == 0:             # using Tp
            if self.Mp <= 0:
                self.zeta = np.nan
                self.wn = np.nan
            else:
                self.zeta = 1.0 / np.sqrt(1.0 + (np.pi / np.log(self.Mp)) ** 2)
                self.wn = np.pi / self.Tp / np.sqrt(1.0 - self.zeta ** 2)
        else: # using Ts
            q = self.Tp / np.pi / self.Ts if self.Ts != 0 else np.nan
            if self.SettlingTimeLimits[0] == 0.01:
                q *= 4.6
            else:
                q *= 4.0
            self.zeta = q / np.sqrt(1.0 + q ** 2) if not np.isnan(q) else np.nan
            self.wn = 4.0 / self.Ts / self.zeta if self.zeta != 0 else np.nan

    def printout(self, verbose=False):
        print(f"omega_n:\t{self.wn:.3f}")
        print(f"zeta   :\t{self.zeta:.3f}")
        print(f"Tr     :\t{self.Tr:.2f}s")
        print(f"Ts     :\t{self.Ts:.2f}s")
        print(f"Mp     :\t{self.Mp:.2f}")
        print(f"Tp     :\t{self.Tp:.2f}s")
        print(f"Yss    :\t{self.Yss:.2f}")
        if verbose:
            return SimpleNamespace(**{
            "Mp": self.Mp,
            "Tr": self.Tr,
            "Ts": self.Ts,
            "Tp": self.Tp,
            "Yss": self.Yss,
            })
        else:
            pass

    def nice_plot(self, ax=None, Tmax=None, Ymax=None, label=None, lc='b'):
        if Ymax is None:
            ylim = (np.floor(np.min(self.y)), np.ceil(10.0 * np.max(self.y)) / 10.0)
            Ymax = np.max(ylim)
        if Tmax is None:
            Tmax = np.max(self.t)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(self.t, self.y, color=lc, label=label)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("Response")
        ax.set_title("Step Response")
        ax.set_ylim(0, Ymax)
        ax.set_xlim(0, Tmax)

        ax.axvline(x=self.Tr_values[0], ymax=0.1 * self.Yss / Ymax, c="r", ls="dashed")
        ax.axvline(x=self.Tr_values[1], ymax=0.9 * self.Yss / Ymax, c="r", ls="dashed")
        ax.axvline(x=self.Ts, ymax=self.Yss / Ymax, c="grey", ls="dashed")
        ax.axvline(ymax=min((self.Yss * (1 + self.Mp)) / Ymax, 1.0), x=self.Tp, c="m", ls="dashed", lw=2)
        ax.axhline(y=(1 + self.SettlingTimeLimits[0]) * self.Yss, xmin=self.Ts / Tmax, c="grey", ls="dashed", lw=1)
        ax.axhline(y=(1 - self.SettlingTimeLimits[0]) * self.Yss, xmin=self.Ts / Tmax, c="grey", ls="dashed", lw=1)
        ax.plot((0, self.Tp), (self.Yss * (1 + self.Mp), self.Yss * (1 + self.Mp)), c="green", ls="dashed", lw=2)
        ax.text(min(self.Tr / 2, Tmax/2), 0.25 * self.Yss, f"Tr = {self.Tr:.2f}", fontsize=SMALL_SIZE)
        ax.text(min(self.Tp, Tmax/2), 0.75 * self.Yss, f"Tp = {self.Tp:.2f}", fontsize=SMALL_SIZE)
        ax.text(min(self.Ts, Tmax/2), 0.5 * self.Yss, f"Ts = {self.Ts:.2f}", fontsize=SMALL_SIZE)
        ax.text(min(self.Tp * 1.1, Tmax/2), min(self.Yss * (1 + self.Mp),0.8*Ymax), f"Mp = {self.Mp:.2f}", fontsize=SMALL_SIZE)
        ax.text(min(self.Ts, Tmax/2), min(self.Yss * 1.1,0.9*Ymax), rf"$e_{{ss}}$ = {1 - self.Yss:.3f}", fontsize=SMALL_SIZE, color="purple")

# -------------------------------
# More utilities
# -------------------------------

def lead_design(G, wc_des = 1, PMdes = 45, verbose=None):

    j = complex(0,1)
    Gf = G(j*wc_des)
    phi_G = wrap_phase_neg(np.angle(Gf)) * r2d # phase of plant at wc (degs)
    PM =  wrap(180.0 + phi_G)

    if verbose:
        print(f"Plant phase {phi_G:.2f}° , PMdes {PMdes:.2f}°, Current PM {PM:.2f}°, Phase required {PMdes - PM:.2f}°")

    phi_required = (PMdes - PM) / r2d  # rads
    zdp = (1.0 - np.sin(phi_required)) / (1.0 + np.sin(phi_required))
    z = float(wc_des * np.sqrt(zdp))
    p = float(z / zdp)

    Gc_lead = ct.tf([1, z], [1, p])
    L = G * Gc_lead
    k_c = 1.0 / np.abs(L(j * wc_des))
    Gc_lead *= k_c

    latex_paragraph = (
        f"The phase of the open-loop transfer function $G(j\\omega_c)$ at the desired crossover frequency "
        f"is $\\phi_G = {phi_G:.2f}^\\circ$. Thus the required phase lead is calculated as "
        f"$\\phi_required = {phi_required * r2d:.2f}^\\circ$. Using the phase lead equation $$\\dfrac{{z}}{{p}} = "
        f"\\dfrac{{1 - \\sin(\\phi_required)}}{{1 + \\sin(\\phi_required)}} = {zdp:.3f}.$$ The zero and pole of the lead "
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

    zl = float(wc / gamma)
    pl = float(zl / gain_inc)
    if verbose:
        latex_paragraph = (
            f"The lag compensator zero at $z_l = {zl:.2f}$ and pole at $p_l = {pl:.3f}$. "
            f"Resulting lag compensator $G^{{lag}}_c(s) = \\dfrac{{s+{zl:.2f}}}{{s+{pl:.3f}}}$"
        )
        return ct.tf([1, zl], [1, pl]), latex_paragraph
    else:
        return ct.tf([1, zl], [1, pl])

def system_type(L, tol=1e-9):
    """
    Number of poles at the origin.
    """
    poles = L.poles()
    return sum(abs(p) < tol for p in poles)

def static_error_constant(L, order):
    """
    Compute Kp (order=0), Kv (order=1), Ka (order=2).
    Returns None if not defined.
    """
    stype = system_type(L)

    if order < stype:
        return 0.0
    if order > stype:
        return None

    s = 1e-6
    val = (s**order) * L(s)
    return float(np.real(val))

def find_Kp(L):
    return static_error_constant(L, order = 0)

def find_Kv(L):
    return static_error_constant(L, order = 1)

def find_Ka(L):
    return static_error_constant(L, order = 2)

def _eval_Gjw(G, omega):
    '''Handle the different formats that a system model might be given'''
    if isinstance(G, (ct.TransferFunction, ct.StateSpace)):
        Gf = G(1j * omega)
    elif callable(G):
        Gf = G(1j * omega)
    else:
        Gf = np.asarray(G, dtype=complex)

    return np.atleast_1d(Gf), np.asarray(omega)

def find_wc(omega, G, mag=1.0, find_all=False, rtol=0.01):
    """
    Find frequency where |G(jω)| is closest to `mag`.

    Parameters
    ----------
    omega : array_like
        Frequency grid (rad/s)
    G : control.TransferFunction, control.StateSpace, callable, or array_like
        System or frequency response
    mag : float, default 1.0
        Target magnitude
    find_all : bool, default False
        If True, return all frequencies within tolerance
    rtol : float, default 0.05
        Relative tolerance (fraction of mag)

    Returns
    -------
    wc : float or ndarray
        Frequency (or frequencies)
    idx : int or ndarray
        Index (or indices) into omega
    """
    Gf, omega = _eval_Gjw(G, omega)

    mag_err = np.abs(np.abs(Gf) - mag)

    if not find_all:
        idx = np.argmin(mag_err)
        return omega[idx], idx

    tol = rtol * mag
    idx_raw = np.where(mag_err <= tol)[0]

    if idx_raw.size == 0:
        return np.array([]), np.array([])

    # group contiguous indices
    groups = np.split(idx_raw, np.where(np.diff(idx_raw) > 1)[0] + 1)

    # choose best representative per group
    idx = np.array([g[np.argmin(mag_err[g])] for g in groups])

    return omega[idx], idx

def find_wpi(omega, G, phi=np.pi, find_all=False, rtol=0.01):
    """
    Find frequency where arg(G(jw)) = phi (degrees)

    If find_all is False
        returns closest match (omega[idx], idx)

    If find_all is True
        returns all crossings (omega_hits, idx_hits)
    """
    Gf, omega = _eval_Gjw(G, omega)

    phase = np.angle(Gf)
    phase_err = np.abs(np.angle(np.exp(1j * (phase - phi))))

    if not find_all:
        idx = np.argmin(phase_err)
        return omega[idx], idx

    tol = rtol * np.pi
    idx_raw = np.where(phase_err <= tol)[0]

    if idx_raw.size == 0:
        return np.array([]), np.array([])

    groups = np.split(idx_raw, np.where(np.diff(idx_raw) > 1)[0] + 1)
    idx = np.array([g[np.argmin(phase_err[g])] for g in groups])

    return omega[idx], idx

def find_PM(omega, G, mag=1.0, wc=None):
    """
    Compute phase margin from frequency response.

    Parameters
    ----------
    omega : array_like
        Frequency grid (rad/s)
    G : control.TransferFunction, control.StateSpace, callable, or array_like
        Open-loop system or frequency response
    mag : float, default 1.0
        Gain crossover magnitude (normally 1)

    Returns
    -------
    PM : float
        Phase margin in degrees (NOT wrapped)
    wc : float
        Gain crossover frequency (rad/s)
    idx : int
        Index into omega corresponding to wc
    """
    if wc is None:
        # find gain crossover
        wc, idx = find_wc(omega, G, mag=mag)

    # evaluate frequency response
    Gf, omega = _eval_Gjw(G, omega)

    # phase at crossover (radians)
    phi = np.angle(Gf[idx])

    # phase margin (degrees)
    PM = 180.0 + np.degrees(phi)

    return PM, wc, idx

def Departure_angle(L,s0,Tol=1e-4):
    '''Departure angle in degrees
    Inputs:
        L - system
        s0 - target point
        Tol - to remove the pole/zero at s0 from the evaluation
    '''
    phi_d = (180+sum([cmath.phase(x) for x in (s0 - L.zeros()) if np.abs(x) > Tol])*r2d \
                -sum([cmath.phase(x) for x in (s0 - L.poles()) if np.abs(x) > Tol])*r2d) % 360 
    return phi_d

def Arrival_angle(L,s0,Tol=1e-4):
    '''Arrival angle in degrees
    Inputs:
        L - system
        s0 - target point
        Tol - to remove the pole/zero at s0 from the evaluation
    '''
    phi_a = (180-sum([cmath.phase(x) for x in (s0 - L.zeros()) if np.abs(x) > Tol])*r2d \
                +sum([cmath.phase(x) for x in (s0 - L.poles()) if np.abs(x) > Tol])*r2d) % 360
    return phi_a

def caption(txt, fig=None, xloc=0.5, yloc=-0.05):
    """
    Add a centered caption to a figure (below the axes).
    If fig is None, uses the current figure.
    """
    if fig is None:
        fig = plt.gcf()
    fig.text(xloc, yloc, txt, ha="center", size=MEDIUM_SIZE, color="blue")

def new_pzmap(G, ax = None, title = None):
    '''PZ map with nicer markers for the poles/zeros
    Inputs:
        G - system
        ax - figure axis, returned if not provided
        title
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        return_ax = True
    else:
        return_ax = False

    ax.plot(np.real(G.poles()), np.imag(G.poles()), "bx", ms=6)
    ax.plot(np.real(G.zeros()), np.imag(G.zeros()), "o", ms=6, 
        markeredgewidth=2,markeredgecolor="r", markerfacecolor="r")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")

    if title is None:
        ax.set_title("Pole-Zero Map")
    else:
        ax.set_title(title)
    ax.grid(True)

    if return_ax:
        return ax

def color_rl(ax, ms=8, lw=1.5):
    '''
    Change RL line colors
    Inputs:
        ms = 8
        lw = 1.5
    '''
    for line in ax.lines:
        if line.get_linestyle() == "-":
            line.set_linewidth(lw)
            line.set_color("blue")
        if line.get_marker() == "x":
            line.set_markersize(ms)
            line.set_color("blue")
        if line.get_marker() == "o":
            line.set_markersize(ms)
            line.set_markerfacecolor("r")
            line.set_markeredgecolor("r")
        if line.get_marker() == "d":
            line.set_markersize(ms)
            line.set_markerfacecolor("g")
            line.set_markeredgecolor("g")

def Read_data(file_name, comments=["#", "F"], cols=[0]):
    return np.loadtxt(file_name, comments=comments, delimiter=",", usecols=cols)

def add_break_info(ax, break_info, dim=None, tol=1e-6):
    '''Add the root locus break in/out info to the plot
    inputs:
    ax: plot
    break_info: from add_break_info
    dim: plot size
    tol: 
    '''
    ymin, ymax = ax.get_ylim()
    ydelta = (ymax - ymin) / 10.0
    xmin, xmax = ax.get_xlim()
    xdelta = xmin + 0.05 * (xmax - xmin)

    if dim is None:
        dim = float(ymax)

    if not break_info:
        return

    for k, bp in enumerate(break_info):

        poles = np.atleast_1d(bp.poles).astype(float)

        # Deduplicate with tolerance
        unique_poles = []
        for p in poles:
            if not any(abs(p - q) <= tol for q in unique_poles):
                unique_poles.append(p)

        # Only display the first unique pole
        pole_str = f"{unique_poles[0]:5.3f}"

        ax.text(
            xdelta,
            dim - (k + 1) * ydelta,
            f"Gain: {bp.K:5.3f} at s = {pole_str}",
            fontsize=8
        )

def near_zero(P, Tol=1e-12):
    '''remove small terms in the num/den of the TF'''
    if not isinstance(P, ct.TransferFunction):
        return P
    num = [x if abs(x) > Tol else 0.0 for x in P.num[0][0]]
    den = [x if abs(x) > Tol else 0.0 for x in P.den[0][0]]
    return ct.tf(num, den)

def log_interp(zz, xx, yy):
    logz = np.log10(zz)
    logx = np.log10(xx)
    logy = np.log10(yy)
    return np.power(10.0, np.interp(logz, logx, logy))

def balred(G, order = None, DCmatch = False, check = False, method = None, Tol=1e-9):
    '''
    Balanced Model reduction using both methods discussed here
    https://web.stanford.edu/group/frg/course_work/CME345/CA-AA216-CME345-Ch6.pdf

    G: System model in TF or SS form

    Free integrators will be removed for reduction and then added back in

    order: dim of system to return
    '''
    from scipy.linalg import solve_continuous_lyapunov, svd

    is_ss = isinstance(G, ct.StateSpace) # in SS form already?
    if is_ss:
        Gin = ct.ss2tf(G)
    else:
        Gin = G

    if method is None:
        method = 0

    G_trimmed = ct.tf(Gin.num[0][0], np.trim_zeros(Gin.den[0][0], "b"))
    number_cut = len(Gin.den[0][0]) - len(G_trimmed.den[0][0])

    Gss = ct.tf2ss(G_trimmed)
    if order is None:
        order = Gss.A.shape[0] - 1
    order -= number_cut
    if order <= 0:
        print("System dimension not correct")
        return

    # check stability
    evals = scipy.linalg.eigvals(Gss.A)
    if np.max(np.real(evals)) > 0:
        print("Algorithm only works for stable systems")
        return

    Wc = solve_continuous_lyapunov(Gss.A, -Gss.B @ Gss.B.T)
    Wo = solve_continuous_lyapunov(Gss.A.T, -Gss.C.T @ Gss.C)
    Wc = (Wc + Wc.T)/2.0
    Wo = (Wo + Wo.T)/2.0
    U = np.linalg.cholesky(Wc)

    if method == 0:
        Z = np.linalg.cholesky(Wo)
        W, Sigma, Vh = svd(U.T @ Z)
        Sigma_sqrt_inv = np.linalg.inv(np.diag(np.sqrt(Sigma)))

        T = Sigma_sqrt_inv @ Vh @ Z.T
        Ti = U @ W @ Sigma_sqrt_inv

    elif method == 1:
        print("Using Method 1 - not recommended for high-order systems")
        from scipy.linalg import eigh # symmetric matrices

        eigvals, K = eigh(U.T @ Wo @ U)
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        K = K[:, idx]

        Sigma = np.sqrt(eigvals)
        Sigma_inv_sqrt = np.diag([1.0/xx for xx in np.sqrt(Sigma)])
        Sigma_sqrt = np.diag([xx for xx in np.sqrt(Sigma)])

        T  = Sigma_sqrt @ K.T @ np.linalg.inv(U)
        Ti = U @ K @ Sigma_inv_sqrt

    Ab = T @ Gss.A @ Ti
    Bb = T @ Gss.B
    Cb = Gss.C @ Ti

    Arr = Ab[:order, :order]
    Brr = Bb[:order, :]
    Crr = Cb[:, :order]
    Drr = Gss.D
    Gr = None
    try:
        Gr = cmat.StateSpace(Arr, Brr, Crr, Drr)
    except Exception:
        Gr = ct.StateSpace(Arr, Brr, Crr, Drr)

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

    def hsv(Wc,Wo):
        from scipy.linalg import sqrtm, svd
        sqrtWc = sqrtm(Wc)
        U, S, Vh = svd(sqrtWc @ Wo @ sqrtWc)
        return np.sqrt(S)           

    if check:
        Wc_bal = solve_continuous_lyapunov(Gr.A, -Gr.B @ Gr.B.T)
        Wo_bal = solve_continuous_lyapunov(Gr.A.T, -Gr.C.T @ Gr.C)
        print("\n Balanced")
        print(Wc_bal)
        print(Wo_bal)
        diff = np.linalg.norm(Wc_bal - Wo_bal)
        if np.abs(diff) < Tol:
            print("Model is balanced\n")
        print("\nTransformed Wc")
        print(T @ Wc @ T.T)
        print("\nTransformed Wo")
        print(np.linalg.inv(T.T) @Wo @ Ti)
        print("\nOriginal model HSV: ",hsv(Wc,Wo))

    Gr = near_zero(ct.ss2tf(Gr)) * ct.tf([1], [1, 0]) ** number_cut
    return ct.tf2ss(Gr) if is_ss else Gr 

def pretty_row_print(X, msg=""):
    print(msg + ", ".join("({0.real:.2f} + {0.imag:.2f}i)".format(x) if np.iscomplex(x) else "{:.3f}".format(x.real) for x in X))

def feedback_ff(G, K, Kff):
    if isinstance(G, (int, float, np.number)):
        G = ct.tf([G], [1])
    elif isinstance(K, ct.StateSpace):
        G = ct.ss2tf(G)
    elif not isinstance(G, ct.TransferFunction):
        raise TypeError("G must be a scalar, TransferFunction, or StateSpace")

    if isinstance(K, (int, float, np.number)):
        K = ct.tf([K], [1])
    elif isinstance(K, ct.StateSpace):
        K = ct.ss2tf(K)
    elif not isinstance(K, ct.TransferFunction):
        raise TypeError("K must be a scalar, TransferFunction, or StateSpace")

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

    return ct.tf(Kff * NGDC + NGNC, DGDC + NGNC)

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

######################################################   
# sympy helpers
#####################################################
def round_constants(expr, ndigits=3):
    return expr.xreplace({
        c: sp.Float(c, ndigits) for c in expr.atoms()
        if c.is_Number and not c.is_Integer
    })


######################################################   
# TF helpers
######################################################
def write_latex_array(X, filename, msgs=None, cols=1, tol=1e-12, decimals=2):
    """
    Write a list/array of (possibly complex) numbers to a LaTeX array
    that can be \\input{} directly.

    Parameters
    ----------
    X : iterable
        Numbers (real or complex)
    filename : str
        Output .tex file
    msgs : str
        latex label
    cols : int
        Number of columns in the array
    tol : float
        Imaginary-part tolerance for treating numbers as real
    """
    def fmt(x):
        xr = np.real(x)
        xi = np.imag(x)
        if abs(xi) < tol:
            return f"{xr:.{decimals}f}"
        else:
            sign = "+" if xi >= 0 else "-"
            return f"({xr:.{decimals}f} {sign} {abs(xi):.{decimals}f}i)"

    entries = [fmt(x) for x in X]

    # split into rows
    rows = [
        entries[i:i+cols]
        for i in range(0, len(entries), cols)
    ]

    with open(filename, "w") as f:
        f.write("\\begin{array}{%s}\n" % ("c" * cols))
        f.write(msgs)
        for r in rows:
            f.write("  " + " & ".join(r) + " \\\\\n")
        f.write("\\end{array}\n")

def show_tf_latex(P, label=None, sigfigs=2, show=None, factor=False, name=None):
    ''' 
    P: system
    label
    show
    factor
    '''
    var = "s"

    if label is None:
        label = f"G({var})"
    if name is not None:
        label = name

    num = np.array(P.num[0][0], dtype=float)
    den = np.array(P.den[0][0], dtype=float)

    if factor:
        Kn, rnum, qnum = factor_poly_real(num)
        Kd, rden, qden = factor_poly_real(den)

        # cancel common real roots
        rnum_c, rden_c = cancel_common_real_roots(rnum, rden)

        num_body = factors_to_latex(rnum_c, qnum, var, sigfigs)
        den_body = factors_to_latex(rden_c, qden, var, sigfigs)

        frac = build_frac_latex_gain_in_numer(Kn, num_body, Kd, den_body, sigfigs)

    else:
        # polynomial (unfactored) form
        num_tex = _poly_to_latex(num, sigfigs=sigfigs, var=var, discrete=False)
        den_tex = _poly_to_latex(den, sigfigs=sigfigs, var=var, discrete=False)
        frac = rf"\displaystyle \frac{{{num_tex}}}{{{den_tex}}}"

    msgs = Math(label + " = " + frac)

    if show:
        display(msgs)

    return msgs


def _num_to_latex(x, sigfigs=4):
    """
    Convert a float to LaTeX-safe numeric string.
    Uses \\times 10^{k} for scientific notation, with clean exponents.
    """
    s = f"{x:.{sigfigs}g}"
    if "e" in s or "E" in s:
        base, exp = s.replace("E", "e").split("e")
        exp = int(exp)   # <-- THIS removes leading zeros
        return rf"{base}\times 10^{{{exp}}}"
    return s

def _sci_to_latex(s):
    """
    Convert '4.4e-06' -> '4.4 \\times 10^{-6}'
    """
    if "e" in s:
        base, exp = s.split("e")
        return rf"{base} \times 10^{{{int(exp)}}}"
    return s

def _poly_to_latex(coefs, sigfigs=4, var="s", discrete=False):
    terms = []
    n = len(coefs)

    for i, val in enumerate(coefs):

        if abs(val) < 1e-12:
            continue

        # sign and magnitude
        sign = "-" if val < 0 else "+"
        mag = abs(val)

        # format once to significant figures
        coeff_str = f"{mag:.{sigfigs}g}"
        coeff_str = _sci_to_latex(coeff_str)

        if discrete:
            power = i
            if power == 0:
                term_body = coeff_str
            else:
                term_body = rf"{'' if coeff_str == '1' else coeff_str}{var}^{{-{power}}}"
        else:
            degree = n - 1 - i
            if degree > 1:
                term_body = rf"{'' if coeff_str == '1' else coeff_str}{var}^{degree}"
            elif degree == 1:
                term_body = rf"{'' if coeff_str == '1' else coeff_str}{var}"
            else:
                term_body = coeff_str

        terms.append((sign, term_body))

    if not terms:
        return "0"

    # first term: suppress leading '+'
    first_sign, first_term = terms[0]
    result = ("" if first_sign == "+" else "-") + first_term

    for sign, term in terms[1:]:
        result += f" {sign} {term}"

    return result

def tf_to_latex(G):
    s = sp.Symbol('s')
    
    # Get numerator and denominator coefficients
    num, den = G.num[0][0], G.den[0][0]

    # Convert to symbolic expressions
    num_poly = sum(np.round(coef,2) * s**i for i, coef in enumerate(reversed(num)))
    den_poly = sum(np.round(coef,2) * s**i for i, coef in enumerate(reversed(den)))

    # Create the LaTeX representation
    G_sym = num_poly / den_poly
    raw_latex = sp.latex(G_sym)
    nice_latex = raw_latex.replace(r"\frac", r"\dfrac")     # force displaystyle fractions
    return nice_latex

def _matrix_to_latex(M, sigfigs=4):
    M = np.atleast_2d(np.array(M, dtype=float))
    rows = []
    for row in M:
        rows.append(
            " & ".join(f"{x:.{sigfigs}g}" for x in row)
        )
    body = r" \\ ".join(rows)
    return r"\begin{bmatrix} " + body + r" \end{bmatrix}"

def show_ss_latex(P, label=None, sigfigs=4, name=None):
    """
    Display a StateSpace system as LaTeX with A, B, C, D matrices.
    """

    if not isinstance(P, ct.StateSpace):
        raise TypeError("Input must be a control.StateSpace object")

    # detect discrete vs continuous
    try:
        is_discrete = ct.isdtime(P)
    except Exception:
        dt = getattr(P, "dt", None)
        is_discrete = isinstance(dt, (int, float)) and dt > 0

    var = "k" if is_discrete else "t"

    # label handling
    if label is None and name is None:
        label = ""
    elif label is None:
        label = name
    else:
        label = label

    A, B, C, D = P.A, P.B, P.C, P.D

    A_tex = _matrix_to_latex(A, sigfigs)
    B_tex = _matrix_to_latex(B, sigfigs)
    C_tex = _matrix_to_latex(C, sigfigs)
    D_tex = _matrix_to_latex(D, sigfigs)

    if is_discrete:
        eqn = (
            r"\begin{aligned}"
            r"x_{k+1} &= " + A_tex + r" x_k + " + B_tex + r" u_k \\ "
            r"y_k &= " + C_tex + r" x_k + " + D_tex + r" u_k"
            r"\end{aligned}"
        )
    else:
        eqn = (
            r"\begin{aligned}"
            r"\dot{x}(t) &= " + A_tex + r" x(t) + " + B_tex + r" u(t) \\ "
            r"y(t) &= " + C_tex + r" x(t) + " + D_tex + r" u(t)"
            r"\end{aligned}"
        )

    if label:
        return Math(label + ":\n" + eqn)
    else:
        return Math(eqn)

# ---------- helpers ----------
if 0:
    def _build_linear_factor_from_root(root, var='s', Tol=1e-9, sigfigs=4):
        # For a root r, factor is (var - r) -> printed as (var + a) where a = -r
        r = complex(root)
        if abs(r.imag) <= Tol:
            a = -float(r.real)
            if abs(a) <= Tol:
                return var  # plain s
            if a > 0:
                return rf"({var} + {fmt(a, sigfigs)})"
            else:
                return rf"({var} - {fmt(abs(a), sigfigs)})"
        else:
            # complex linear factor (unlikely to print alone for real polynomials)
            real = fmt(-r.real, sigfigs)
            imag = fmt(-r.imag, sigfigs)
            sign = '+' if r.imag < 0 else '-'
            # show as (s - (a + jb)) with j sign adjusted: (s - (ar + j ai))
            return rf"({var} - ({fmt(r.real,sigfigs)} {sign} {fmt(abs(r.imag),sigfigs)}j))"

def fmt(x, sigfigs=4, tol=1e-10):
    x = float(x)
    if abs(x) < tol:
        return f"{0:.{sigfigs}f}"
    return f"{x:.{sigfigs}f}"

#def fmt(x, sigfigs=4):
#    # format float for LaTeX: avoid scientific notation for moderate values
#    return np.format_float_positional(float(x), precision=sigfigs, trim='-')

def build_frac_latex(Kn, num_body, Kd, den_body, sigfigs=4, tol=1e-8):
    #fmt = lambda x: np.format_float_positional(x, precision=sigfigs, trim='-')

    if num_body == "1":
        num_body = None
    if den_body == "1":
        den_body = None

    K = Kn / Kd

    if den_body is None and num_body is None:
        return rf"\displaystyle {fmt(K, sigfigs)}"

    if den_body is None:
        if abs(K - 1) < tol:
            return rf"\displaystyle {num_body}"
        return rf"\displaystyle {fmt(K, sigfigs)}\,{num_body}"

    if num_body is None:
        return rf"\displaystyle {fmt(K, sigfigs)}\,\frac{{1}}{{{den_body}}}"

    if abs(K - 1) < tol:
        return rf"\displaystyle \frac{{{num_body}}}{{{den_body}}}"

    return rf"\displaystyle {fmt(K)}\,\frac{{{num_body}}}{{{den_body}}}"

def factors_to_latex(real_roots, quads, var="s", sigfigs=4, tol=1e-8):
    #fmt = lambda x: np.format_float_positional(x, precision=sigfigs, trim='-')
    parts = []

    for r in sorted(real_roots):
        a = -r
        if abs(a) < tol:
            parts.append(var)
        elif a > 0:
            #parts.append(f"({var}+{fmt(a)})")
            parts.append(f"({var}+{fmt(a, sigfigs)})")
        else:
            #parts.append(f"({var}-{fmt(abs(a))})")
            parts.append(f"({var}-{fmt(abs(a), sigfigs)})")

    for B, C in quads:
        Bs = fmt(B, sigfigs)
        Cs = fmt(C, sigfigs)
        parts.append(f"({var}^2+{Bs}{var}+{Cs})")

    return "1" if not parts else "".join(parts)

# ---------- cancellation helper ----------
def cancel_common_roots(K_num, roots_num, K_den, roots_den, tol=1e-6):
    """
    Remove common roots between num and den (within tol), update K_num/K_den accordingly.
    Returns K_num_new, roots_num_new, K_den_new, roots_den_new
    """
    num_roots = roots_num.copy()
    den_roots = roots_den.copy()
    used_den = [False]*len(den_roots)
    remaining_num = []
    for r in num_roots:
        found = False
        for j, rd in enumerate(den_roots):
            if not used_den[j] and abs(r - rd) <= tol:
                # cancel this root pair
                used_den[j] = True
                found = True
                break
        if not found:
            remaining_num.append(r)
    remaining_den = [rd for j,rd in enumerate(den_roots) if not used_den[j]]
    # If canceled roots, adjust scalar gain by multiplying factor (den/numer) from polynomial values.
    # Simpler: leave K_num/K_den unchanged (they are leading coeffs). Cancelling roots does not change K_total.
    return K_num, remaining_num, K_den, remaining_den

# ---------- build fraction latex ----------
def build_fraction_latex_from_roots(K_num, roots_num, K_den, roots_den,
                                    var='s', sigfigs=4, Tol=1e-9, cancel=False):
    # optionally cancel common roots
    if cancel:
        K_num, roots_num, K_den, roots_den = cancel_common_roots(K_num, roots_num, K_den, roots_den, tol=1e-6)

    # build bodies from numeric roots (must be symmetric with factor builder)
    num_parts = []
    for r in sorted(roots_num, key=lambda z: (round(z.real,8), round(z.imag,8))):
        num_parts.append(_build_linear_factor_from_root(r, var=var, Tol=Tol, sigfigs=sigfigs))
    den_parts = []
    for r in sorted(roots_den, key=lambda z: (round(z.real,8), round(z.imag,8))):
        den_parts.append(_build_linear_factor_from_root(r, var=var, Tol=Tol, sigfigs=sigfigs))

    num_body = "1" if len(num_parts) == 0 else "".join(num_parts)
    den_body = "1" if len(den_parts) == 0 else "".join(den_parts)

    # compute K_total
    if abs(K_den) < 1e-16:
        K_total = np.inf
    else:
        K_total = float(K_num) / float(K_den)
    K_tex = fmt(K_total, sigfigs)

    # clean
    def clean(b):
        if b is None:
            return None
        b = str(b).strip()
        return None if b in ("", "1") else b

    nb = clean(num_body)
    db = clean(den_body)

    # assemble with safe spacing and +/- handling
    if nb is None and db is None:
        if np.isfinite(K_total):
            return rf"\displaystyle {K_tex}"
        else:
            return r"\displaystyle 0"

    if db is None:
        if abs(K_total - 1) < Tol:
            return rf"\displaystyle {nb}"
        elif abs(K_total + 1) < Tol:
            return rf"\displaystyle -{nb}"
        else:
            return rf"\displaystyle {K_tex}\,{nb}"

    if nb is None:
        if abs(K_total - 1) < Tol:
            return rf"\displaystyle \frac{{1}}{{{db}}}"
        elif abs(K_total + 1) < Tol:
            return rf"\displaystyle -\frac{{1}}{{{db}}}"
        else:
            return rf"\displaystyle {K_tex}\,\frac{{1}}{{{db}}}"

    # both present
    if abs(K_total - 1) < Tol:
        return rf"\displaystyle \frac{{{nb}}}{{{db}}}"
    elif abs(K_total + 1) < Tol:
        return rf"\displaystyle -\frac{{{nb}}}{{{db}}}"
    else:
        return rf"\displaystyle {K_tex}\,\frac{{{nb}}}{{{db}}}"


def cancel_common_real_roots(rnum, rden, tol=1e-6):
    rnum = list(rnum)
    rden = list(rden)

    rnum_out = []
    rden_used = [False]*len(rden)

    for rn in rnum:
        cancelled = False
        for j, rd in enumerate(rden):
            if not rden_used[j] and abs(rn - rd) < tol:
                rden_used[j] = True
                cancelled = True
                break
        if not cancelled:
            rnum_out.append(rn)

    rden_out = [rd for j, rd in enumerate(rden) if not rden_used[j]]
    return rnum_out, rden_out

def build_frac_latex_gain_in_numer(Kn, num_body, Kd, den_body, sigfigs=4, Tol=1e-9):
    """
    Build LaTeX for:
        (Kn/Kd) * (num_body / den_body)
    with the net gain placed INSIDE the numerator.

    Rules:
      - gain appears exactly once
      - num_body or den_body == "1" is suppressed
      - K≈1 omitted, K≈-1 shown as minus sign
    """
    #fmt = lambda x: np.format_float_positional(float(x), precision=sigfigs, trim='-')

    def clean(body):
        if body is None:
            return None
        b = str(body).strip()
        return None if b in ("", "1") else b

    nb = clean(num_body)
    db = clean(den_body)

    # net gain
    K = Kn / Kd
    Ktex = fmt(K, sigfigs)

    # ---- cases ----

    # pure scalar
    if nb is None and db is None:
        return rf"\displaystyle {Ktex}"

    # numerator only
    if db is None:
        if abs(K - 1) < Tol:
            return rf"\displaystyle {nb}"
        elif abs(K + 1) < Tol:
            return rf"\displaystyle -{nb}"
        else:
            return rf"\displaystyle {Ktex}\,{nb}"

    # denominator only
    if nb is None:
        if abs(K - 1) < Tol:
            return rf"\displaystyle \frac{{1}}{{{db}}}"
        elif abs(K + 1) < Tol:
            return rf"\displaystyle -\frac{{1}}{{{db}}}"
        else:
            return rf"\displaystyle \frac{{{Ktex}}}{{{db}}}"

    # full fraction
    if abs(K - 1) < Tol:
        return rf"\displaystyle \frac{{{nb}}}{{{db}}}"
    elif abs(K + 1) < Tol:
        return rf"\displaystyle -\frac{{{nb}}}{{{db}}}"
    else:
        return rf"\displaystyle \frac{{{Ktex}\,{nb}}}{{{db}}}"

def factor_poly_real(coeffs, tol=1e-6):
    """
    coeffs: highest to lowest
    returns:
        K          : leading coefficient
        real_roots : list of real roots
        quads      : list of (B, C) for s^2 + B s + C
    """
    coeffs = np.asarray(coeffs, dtype=float)
    K = coeffs[0]
    roots = np.roots(coeffs)

    used = np.zeros(len(roots), dtype=bool)
    real_roots = []
    quads = []

    for i, r in enumerate(roots):
        if used[i]:
            continue

        imag_tol = tol * (1.0 + abs(r.real))
        if abs(r.imag) <= imag_tol:
            real_roots.append(r.real)
            used[i] = True
            continue

        # otherwise, try to form a conjugate quadratic
        for j in range(i+1, len(roots)):
            if not used[j] and abs(roots[j] - np.conj(r)) <= imag_tol:
                used[i] = used[j] = True
                a = r.real
                b = abs(r.imag)
                quads.append((-2*a, a*a + b*b))
                break

    return K, real_roots, quads



def pid(Kp = 0, Ki = 0, Kd = 0):
    '''return tf form of a PID controller given Kp,Ki,Kd'''
    s = ct.tf((1,0),(1))
    return ct.tf(Kp,1) + Ki/s + Kd*s

def nyquist(*args, **kwargs):
    """
    Wrapper around control.nyquist_plot that always suppresses title
    unless explicitly overridden.
    """
    kwargs.setdefault("title", "")
    return ct.nyquist_plot(*args, **kwargs)

def write_latex_constants(S0, filename="./figs/constants.tex", idname=None, fmt="%.2f"):
    '''
    consts = {"wn": wn,
        "zeta": zeta,
        "c1": c1,
        "c2": c2}
    filename
    idname
    fmt="%.2f"
    '''
    def sanitize_letters(s):
        # allow letters only (TeX control sequence safe)
        return re.sub(r"[^A-Za-z]", "", s)

    suffix = ""
    if idname:
        suffix = sanitize_letters(idname).capitalize()

    with open(filename, "w") as f:
        f.write("% Auto-generated by Python. Do not edit.\n")
        for name, val in S0.items():
            macro = sanitize_letters(name) + suffix
            f.write(r"\def\%s{%s}" % (macro, fmt % val) + "\n")

def write_tf_latex(G, filename, label, sigfigs=4, factor=None):
    ''' G filename sigfigs'''
    tex = show_tf_latex(G,sigfigs=sigfigs,factor=factor, show=False)._repr_latex_()
    tex = tex.strip("$")

    # remove everything up to the first '='
    if "=" in tex:
        tex = tex.split("=", 1)[1].strip()

    with open(filename, "w") as f:
        f.write(r"\[" + "\n")
        f.write(label + " = " + tex + "\n")
        f.write(r"\]" + "\n")


def normalize_tf(G):
    '''factor out non-unity gain for leading coefficient of the denominator'''
    if isinstance(G, ct.StateSpace):
        G = ct.ss2tf(G)

    num,den = G.num[0][0],G.den[0][0]
    num = num/den[0]
    den = den/den[0]
    return ct.tf(num,den)

def find_double_real_poles(real_poles, tol=1e-5):
    """
    Identify real poles that occur more than once (within tolerance)
    and return one representative value per location.
    """
    real_poles = np.asarray(real_poles, dtype=float)
    real_poles = np.sort(real_poles)

    doubles = []
    i = 0
    n = len(real_poles)

    while i < n - 1:
        if abs(real_poles[i+1] - real_poles[i]) < tol:
            # representative value (average of cluster)
            cluster = [real_poles[i]]
            j = i + 1
            while j < n and abs(real_poles[j] - real_poles[i]) < tol:
                cluster.append(real_poles[j])
                j += 1
            doubles.append(np.mean(cluster))
            i = j
        else:
            i += 1

    return doubles

def U(t):
    """
    Unit step function.

    Parameters
    ----------
    t : array_like

    Returns
    -------
    ndarray
    """
    t = np.asarray(t)
    u = np.zeros_like(t)
    u[t >= 0] = 1
    return u


def legend_best_combined(ax, candidates=None,
                         w_text=10.0, w_data=1.0,
                         **legend_kwargs):
    """
    Place legend minimizing overlap with both text (incl. AnchoredText)
    and plotted data, similar to loc='best' but text aware.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    candidates : iterable of str
        Legend locations to consider.
    w_text : float
        Weight for text overlap (large).
    w_data : float
        Weight for data overlap (smaller).
    **legend_kwargs :
        Passed to ax.legend().

    Returns
    -------
    legend : matplotlib.legend.Legend
    """
    if candidates is None:
        candidates = [
            "upper right",
            "upper left",
            "lower left",
            "lower right",
        ]

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # ---- text / annotation bboxes ----
    text_bboxes = []

    for t in ax.texts:
        text_bboxes.append(t.get_window_extent(renderer))

    for a in ax.artists:
        if hasattr(a, "get_window_extent"):
            try:
                text_bboxes.append(a.get_window_extent(renderer))
            except Exception:
                pass

    # ---- data bboxes (lines, patches, collections) ----
    data_bboxes = []

    for line in ax.lines:
        data_bboxes.append(line.get_window_extent(renderer))

    for p in ax.patches:
        data_bboxes.append(p.get_window_extent(renderer))

    for c in ax.collections:
        data_bboxes.append(c.get_window_extent(renderer))

    best_loc = None
    best_cost = np.inf

    for loc in candidates:
        leg = ax.legend(loc=loc, **legend_kwargs)
        fig.canvas.draw()
        leg_bbox = leg.get_window_extent(renderer)

        text_overlap = sum(leg_bbox.overlaps(bb) for bb in text_bboxes)
        data_overlap = sum(leg_bbox.overlaps(bb) for bb in data_bboxes)

        cost = w_text * text_overlap + w_data * data_overlap

        if cost < best_cost:
            best_cost = cost
            best_loc = loc

        leg.remove()

    return ax.legend(loc=best_loc, **legend_kwargs)
