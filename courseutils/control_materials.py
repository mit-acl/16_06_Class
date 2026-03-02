"""
control_materials.py

Control utilities for 16.06.
All environment/setup is opt-in via setup_environment().
"""

__version__ = "16.06-0.5"

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import control as ct
import control.matlab as cmat

import importlib.util
from dataclasses import dataclass
from typing import List
from IPython.display import Math, display, Markdown, Latex

import scipy.linalg
from scipy.linalg import solve_continuous_lyapunov, svd, sqrtm, cholesky, eigvals, eigh # symmetric matrices
from scipy.signal import residue
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

    Tol - closeness to being real
    standard_locus - 180 or 0 deg locus
    Tol_max - limits how large the gains found will be
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
        Num, Den = ct.tfdata(L)
        Num = np.atleast_1d(np.squeeze(Num))
        Den = np.atleast_1d(np.squeeze(Den))

        npoles = len(Den)
        nzeros = len(Num)
        n_add = int(npoles - nzeros)
        L_num_add = np.pad(Num, (n_add, 0), "constant", constant_values=(0,))

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
                phi_temp = Den + kk * L_num_add

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
        print("failed to find breakin/out points:", e)

    Krange = [float(k) for k in Krange]

    if verbose:
        return Krange, break_info
    else:
        return Krange

def RL_COM(L, standard_locus=True):
    """
    Return center of mass and angle (in degs) for root locus asymptotes.
    """
    p = np.asarray(L.poles(), dtype=complex)
    z = np.asarray(L.zeros(), dtype=complex)
    Num_excess_poles = len(p) - len(z)

    if Num_excess_poles <= 0:
        return None, None

    CoM = np.real(sum(p) - sum(z)) / Num_excess_poles

    # Asymptote angles
    k = np.arange(Num_excess_poles)
    if standard_locus:
        angles = (2*k + 1) * 180.0 / Num_excess_poles # angles = (2k+1) * 180 / Num_excess_poles
    else:
        angles = k * 360.0 / Num_excess_poles         # 360/Num_excess_poles spacing starting at 0

    return CoM, angles

def plot_rl_asymptotes(ax, com, angles_deg, rmax=10, **kwargs):
    """
    Plot root locus asymptotes.
    ax         : matplotlib axis
    com        : centroid (real scalar)
    angles_deg : iterable of angles in degrees
    rmax       : length of asymptotes
    kwargs     : passed to ax.plot (e.g. color, linewidth)
    """
    rho = np.linspace(0, rmax, 2)
    for ang in angles_deg:
        s = com + rho * np.exp(1j * np.deg2rad(ang))
        ax.plot(s.real,s.imag,linestyle="--",**kwargs)

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

def Root_Locus_design_cancel(G, s_target=complex(-1, 2), s_cancel=-1, verbose=False):
    """
    Root locus lead design by cancelling/placing pole at s_cancel to get CL poles at s_target.
    Returns Gc, Gcl_poles()
    """

    info = None
    Gczeros = np.array([np.real(s_cancel)])

    # phase from G at target pole
    phi_fromG = wrap_phase_neg(phase_at_freq(G, s_target))
    phi_fromG = float(np.atleast_1d(phi_fromG)[0])

    # phase from cancelling zero
    phi_from_Gc_zero = float(sum(np.angle(s_target - z) for z in np.atleast_1d(np.real(s_cancel))) * r2d)

    # required phase to satisfy angle condition
    phi_required = -wrap_phase_neg(-180.0 - (phi_fromG + phi_from_Gc_zero))

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
        latex_paragraph = (
            rf"The phase of the open-loop plant at the target pole $s_{{target}}$ is "
            rf"$\phi_G = {phi_fromG:.2f}^\circ$. A compensator zero placed at $z = {Gczeros[0]:.2f}$ contributes an additional phase of "
            rf"$\phi_{{Gc,z}} = {phi_from_Gc_zero:.2f}^\circ$. Enforcing the root-locus angle condition "
            rf"$\angle G(s_{{target}}) + \angle G_c(s_{{target}}) = -180^\circ$, the (absolute value of the) required phase from the compensator pole is "
            rf"$$\phi_{{required}} = \arctan\!\left(\frac{{Im\{{s_{{target}}\}}}}{{Re\{{s_{{target}}\}} + P}}\right) \qquad \Rightarrow \qquad "
            rf"P = -Re\{{s_{{target}}\}} + \frac{{Im\{{s_{{target}}\}}}}{{\tan(\phi_{{required}})}}$$ "            
            rf"so that, with $\phi_{{required}} = {phi_required:.2f}$ degs, the compensator pole is at $P = {P:.3f}$. This yields the compensator $G_c(s) = k_c\dfrac{{s + {-Gczeros[0]:.2f}}}{{s + {P:.2f}}}$, and" 
            rf" the compensator gain is selected to satisfy $$|G(s_{{target}})G_c(s_{{target}})| = 1,$$ resulting in a gain of $k_c = {Gain:.3f}$, so that"
            rf"$$G_c(s) = {Gain:.2f}\dfrac{{s + {-Gczeros[0]:.2f}}}{{s + {P:.2f}}}$$" 
        )
        info = SimpleNamespace(
            phi_from_G=phi_fromG,
            phi_from_Gc=phi_from_Gc_zero,
            phi_required=phi_required,
            info=latex_paragraph,
        )

    return Gc, Gcl.poles(), info

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

def Root_Locus_design_ratio(G, s_target=complex(-1, 2), gamma=10, verbose=False, indx=None):
    """
    Root locus design using zero/pole ratio gamma.
    Returns Gc, Gcl_poles()
    """

    sigma_t = -s_target.real
    omega_t = s_target.imag

    phi_fromG = wrap_phase_neg(phase_at_freq(G,s_target))
    phi_deg = -180 - phi_fromG 
    tphi = np.tan(phi_deg/r2d)

    # Quadratic coefficients Az^2 + Bz + C = 0
    A = gamma 
    B = -(gamma + 1) * (sigma_t) - (gamma - 1)*omega_t/tphi
    C =  (sigma_t**2 + omega_t**2)
    # Solve quadratic
    roots = np.roots([A, B, C])

    # Choose physically meaningful root (positive real z)
    z_candidates = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > 0]
    if not z_candidates:
        raise ValueError("No physically valid zero location found")

    if indx is None:
        indx = 0
    z = z_candidates[indx]
    p = gamma * z
    # Gain to satisfy magnitude condition at s_target
    Gc = ct.tf([1,z],[1,p])
    k = 1/np.abs(Gc(s_target)*G(s_target))
    Gc = k*Gc

    L = G * Gc
    #close loop
    Gcl = ct.feedback(L)

    # Build LaTeX
    latex_paragraph = (
        rf"The compensator must contribute ${phi_deg:.0f}^\circ$ "
        rf"of phase at $s_{{\text{{target}}}}$, giving"
        r"$$"
        rf"\tan^{{-1}}\!\left(\dfrac{{{omega_t:.3f}}}{{z - ({sigma_t:.3f})}}\right)"
        r" - "
        rf"\tan^{{-1}}\!\left(\dfrac{{{omega_t:.3f}}}{{\gamma z - ({sigma_t:.3f})}}\right)"
        rf" = {phi_deg:.0f}^\circ"
        r"$$"
        r"Using $\tan(A-B)=\dfrac{\tan A-\tan B}{1+\tan A\tan B}$ we obtain"
        r"$$"
        rf"\dfrac{{{omega_t:.3f}}}{{z - ({sigma_t:.3f})}}"
        r" - "
        rf"\dfrac{{{omega_t:.3f}}}{{\gamma z - ({sigma_t:.3f})}}"
        rf" = \tan({phi_deg:.0f}^\circ)\left("
        r"1 + "
        rf"\dfrac{{{omega_t:.3f}}}{{z - ({sigma_t:.3f})}}"
        rf"\dfrac{{{omega_t:.3f}}}{{\gamma z - ({sigma_t:.3f})}}"
        r"\right)"
        r"$$"
        rf"which, with $\gamma = {gamma:.3f}$ simplifies to"
        r"$$"
        rf"{A:.3f} z^2 {B:+.3f} z {C:+.3f} = 0"
        r"$$"
        rf"Thus $z = {z:.3f}$ and $p = {p:.3f}$. "
        rf"The required gain is $k = {k:.2f}$, yielding"
        r"$$"
        rf"G_c(s) = {k:.2f}\dfrac{{s+{z:.3f}}}{{s+{p:.3f}}}."
        r"$$"
    )

    if verbose:
        return Gc, Gcl.poles(), SimpleNamespace(**{
        "phi_from_G": phi_fromG,
        "phi_required": phi_deg,
        "info": latex_paragraph,
        "zero_candidates": z_candidates,
    })
    else:
        return Gc, Gcl.poles()

def Root_Locus_design_PD(G, s_target=complex(-1, 2), verbose=False, Tol=1e-5):
    """
    PD design to place CL poles at s_target.
    Returns (Gc, Gcl_poles).
    """
    phi_fromG = wrap_phase_neg(phase_at_freq(G,s_target))
    phi_required_deg = (-180 - phi_fromG)%360

    z = s_target.imag / np.tan(phi_required_deg/r2d) - s_target.real

    # without gain
    Gc = ct.tf((1, z), 1)
    L = G*Gc
    if np.abs(np.imag(L(s_target))) < Tol:
        Gain = -1.0 / np.real(L(s_target))
    else:
        print("Possible RL error")
        Gain = -1.0 / np.abs(L(s_target))

    # apply gain
    Gc *= Gain
    L *= Gain
    Gcl = ct.feedback(L)

    if verbose:
        return Gc, Gcl.poles(), SimpleNamespace(**{
        "phi_from_G": phi_fromG,
        "phi_required": phi_required_deg,
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
    Compute Tol=2% settling time.
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
        #ax.set_aspect('equal', adjustable='box')

        ax.axvline(x=self.Tr_values[0], ymax=0.1 * self.Yss / Ymax, c="r", ls="dashed")
        ax.axvline(x=self.Tr_values[1], ymax=0.9 * self.Yss / Ymax, c="r", ls="dashed")
        ax.axvline(x=self.Ts, ymax=self.Yss / Ymax * (1-self.SettlingTimeLimits[0]), c="grey", ls="dashed")
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

def hf_gain(G):
    num, den = ct.tfdata(G)
    num = np.atleast_1d(np.squeeze(num))
    den = np.atleast_1d(np.squeeze(den))

    deg_num = len(num) - 1
    deg_den = len(den) - 1

    if deg_num < deg_den:
        return 0.0
    elif deg_num > deg_den:
        return np.inf
    else:
        return num[0] / den[0]

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
    phi_d = (180+sum([np.angle(x, deg=True) for x in (s0 - L.zeros()) if np.abs(x) > Tol]) \
                -sum([np.angle(x, deg=True) for x in (s0 - L.poles()) if np.abs(x) > Tol])) % 360 
    return phi_d

def Arrival_angle(L,s0,Tol=1e-4):
    '''Arrival angle in degrees
    Inputs:
        L - system
        s0 - target point
        Tol - to remove the pole/zero at s0 from the evaluation
    '''
    phi_a = (180-sum([np.angle(x, deg=True) for x in (s0 - L.zeros()) if np.abs(x) > Tol]) \
                +sum([np.angle(x, deg=True) for x in (s0 - L.poles()) if np.abs(x) > Tol])) % 360
    return phi_a

def caption(txt, fig=None, xloc=0.5, yloc=-0.05):
    """
    Add a centered caption to a figure (below the axes).
    If fig is None, uses the current figure.
    """
    if fig is None:
        fig = plt.gcf()
    fig.text(xloc, yloc, txt, ha="center", size=MEDIUM_SIZE, color="blue")

def new_pzmap(G, ax = None, title = None, ms = 6):
    '''PZ map with nicer markers for the poles/zeros
    Inputs:
        G - system
        ax - figure axis, returned if not provided
        title
    '''
    return_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        return_ax = True
    
    ax.plot(np.real(G.poles()), np.imag(G.poles()), "bx", ms=ms,zorder=1)
    ax.plot(np.real(G.zeros()), np.imag(G.zeros()), "o", ms=ms, 
        markeredgewidth=2,markeredgecolor="r", markerfacecolor="r",zorder=10)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")

    if title is None:
        ax.set_title("Pole-Zero Map")
    else:
        ax.set_title(title)
    ax.grid(True)

    if return_ax:
        return ax

def color_rl(ax, ms=6, lw=1.75, verbose=False):
    """
    Change RL line colors and stacking order.

    unique = cm.color_rl(ax,verbose=True)
    ax.legend(unique.values(), unique.keys())

    """

    # stacking policy
    order = {
        "branch": 2,
        "pole": 4,
        "zero": 10,
        "s0": 8,
        "scl": 20,
    }

    for line in ax.lines:

        # suppress rlocus legend duplication
        if line.get_label().startswith("sys"):
            line.set_label("_nolegend_")

        x = np.asarray(line.get_xdata())
        y = np.asarray(line.get_ydata())

        # Skip completely empty lines (defensive)
        if x.size == 0 or y.size == 0:
            continue

        is_vertical = np.allclose(x, x[0])
        is_horizontal = np.allclose(y, y[0])

        # root locus branches
        if (line.get_linestyle() == "-" and line.get_marker() == "None" 
            and not (is_vertical or is_horizontal)):
            line.set_linewidth(lw)
            line.set_color("blue")
            line.set_zorder(order["branch"])

        # open loop poles
        elif line.get_marker() == "x":
            line.set_markersize(ms)
            line.set_markeredgecolor("blue")
            line.set_zorder(order["pole"])

        # open loop zeros
        elif line.get_marker() == "o":
            line.set_markersize(ms)
            line.set_markerfacecolor("r")
            line.set_markeredgecolor("r")
            line.set_zorder(order["zero"])

        # s0 diamonds
        elif line.get_marker() == "d":
            line.set_markersize(ms)
            line.set_markerfacecolor("m")
            line.set_markeredgecolor("m")
            line.set_zorder(order["s0"])

        # scl squares
        elif line.get_marker() == "s":
            line.set_markersize(int(ms * 0.75))
            line.set_markerfacecolor("c")
            line.set_markeredgecolor("c")
            line.set_zorder(order["scl"])

    if verbose:
        handles, labels = ax.get_legend_handles_labels()
        unique = {}
        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = h
        return unique

def Read_data(file_name, comments=["#", "F"], cols=[0]):
    return np.loadtxt(file_name, comments=comments, delimiter=",", usecols=cols)

def add_break_info(ax, break_info, dim=None, tol=1e-6, delta=None, sigfigs=3):
    """Add the root locus break in/out info to the plot"""

    if delta is None:
        xrel = 0.05
        yrel = 0.1
    else:
        xrel, yrel = delta

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()

    xdelta = xmin + xrel * (xmax - xmin)

    if dim is None:
        dim = float(ymax)

    if not break_info:
        return

    # -------------------------------------------------
    # Deduplicate entire BreakPoint objects
    # -------------------------------------------------
    unique_bp = []
    for bp in break_info:
        poles = np.atleast_1d(bp.poles).astype(float)

        is_new = True
        for ubp in unique_bp:
            upoles = np.atleast_1d(ubp.poles).astype(float)

            if np.isclose(bp.K, ubp.K, atol=tol) and \
               np.allclose(poles, upoles, atol=tol):
                is_new = False
                break

        if is_new:
            unique_bp.append(bp)

    # -------------------------------------------------
    # Now display only unique breakpoints
    # -------------------------------------------------
    for k, bp in enumerate(unique_bp):

        poles = np.atleast_1d(bp.poles).astype(float)
        pole_str = f"{poles[0]:.{sigfigs}f}"

        ax.text(xdelta,
            ymin + (k + 1) * yrel * (ymax - ymin),
            f"Gain: {bp.K:.{sigfigs}f} at s = {pole_str}",
            fontsize=8)

def near_zero(P, Tol=1e-12):
    '''remove small terms in the num/den of the TF'''
    if not isinstance(P, ct.TransferFunction):
        return P

    num, den = ct.tfdata(P)
    num = np.atleast_1d(np.squeeze(num))
    den = np.atleast_1d(np.squeeze(den))

    num = [x if abs(x) > Tol else 0.0 for x in num]
    den = [x if abs(x) > Tol else 0.0 for x in den]
    return ct.tf(num, den)

def log_interp(zz, xx, yy):
    logz = np.log10(zz)
    logx = np.log10(xx)
    logy = np.log10(yy)
    return np.power(10.0, np.interp(logz, logx, logy))

def pick_model_order_from_hsvs(sigmas,method="combined",energy_thresh=0.95,error_thresh=None):
    """
    Choose truncation order r from sorted HSVs (descending).
    method: "gap" | "energy" | "error" | "combined"
    - gap: find the largest ratio sigma_i / sigma_{i+1}
    - energy: cumulative energy fraction >= energy_thresh
    - error: choose r such that 2 * sum_{i>r} sigma_i <= error_thresh
    - combined: prefer gap if large, else energy, else error if provided
    Returns r (int, number of retained states).
    """
    s = np.asarray(sigmas, dtype=float)
    if s.ndim != 1:
        s = s.ravel()
    n = len(s)
    if n == 0:
        return 0

    # energy approach
    total = s.sum()
    cum = np.cumsum(s)
    if total == 0:
        r_energy = 0
    else:
        r_energy = int(np.searchsorted(cum / total, energy_thresh) + 1)
        r_energy = min(max(r_energy, 0), n)

    # gap approach: largest ratio sigma_i / sigma_{i+1}
    ratios = s[:-1] / (s[1:] + 1e-20)
    if len(ratios) == 0:
        r_gap = n
    else:
        idx = np.argmax(ratios)
        # only accept gap if ratio is significant
        if ratios[idx] > 10:   # heuristic threshold; tune as needed
            r_gap = idx + 1
        else:
            r_gap = None

    # error bound approach
    if error_thresh is not None:
        tail_cumsum = np.cumsum(s[::-1])[::-1]  # tail sum from i to end
        # want smallest r with 2*sum_{i>r} sigma_i <= error_thresh
        r_err = n
        for r in range(n+1):
            tail = tail_cumsum[r] if r < n else 0.0
            if 2.0 * tail <= error_thresh:
                r_err = r
                break
    else:
        r_err = None

    if method == "gap":
        return r_gap if r_gap is not None else r_energy
    if method == "energy":
        return r_energy
    if method == "error":
        return r_err if r_err is not None else n
    # combined: prefer gap if found, else energy, else error if small
    if r_gap is not None:
        return r_gap
    if r_energy > 0:
        return r_energy
    if r_err is not None:
        return r_err
    return n

def hsv(Wc, Wo, tol_chol=1e-12):
    """
    Return Hankel singular values robustly from controllability Wc and observability Wo gramians.
    Tries Cholesky-based method first (more stable), falls back to sqrtm+SVD.
    """
    # Try Cholesky-based approach if Wc and Wo are (numerically) SPD
    try:
        Rc = cholesky(Wc, lower=False)   # Rc.T @ Rc = Wc
        Ro = cholesky(Wo, lower=False)
        # compute SVD of Ro @ Rc.T  (or Rc @ Ro.T) - both give same singular values
        U, s, Vt = svd(Ro @ Rc.T)
        return s  # these are the HSVs
    except Exception:
        # fallback: sqrtm( Wc ) * Wo * sqrtm(Wc)
        # use sqrtm then SVD of the symmetric product
        S = sqrtm(Wc) @ Wo @ sqrtm(Wc)
        # S should be symmetric; do eig or svd
        vals = svd(S, compute_uv=False)
        # eigenvalues may be tiny negative due to numerical noise; clip
        vals = np.clip(vals, 0.0, None)
        return np.sqrt(vals)

def balred(G, order = None, DCmatch = False, check = False, method = None, Tol=1e-9):
    '''
    Balanced Model reduction using both methods discussed here
    https://web.stanford.edu/group/frg/course_work/CME345/CA-AA216-CME345-Ch6.pdf

    G: System model in TF or SS form

    Free integrators will be removed for reduction and then added back in

    order: dim of system to return
    '''
    is_ss = isinstance(G, ct.StateSpace) # in SS form already?
    if is_ss:
        Gin = ct.ss2tf(G)
    else:
        Gin = G

    if method is None:
        method = 0

    # remove poles at origin, which are added back in at the end
    num, den = ct.tfdata(Gin)
    num = np.atleast_1d(np.squeeze(num))
    den = np.atleast_1d(np.squeeze(den))

    G_trimmed = ct.tf(num, np.trim_zeros(den, "b"))
    num_trimmed, den_trimmed = ct.tfdata(G_trimmed)
    num_trimmed = np.squeeze(num_trimmed)
    den_trimmed = np.squeeze(den_trimmed)

    number_cut = len(den) - len(den_trimmed) # how many poles at origin to add back in
    if number_cut > 0:
        print(f"\nNumber of free integrators removed (will be added back in): {number_cut :2d}")

    # now operate on the SS model of the trimmed system
    Gss = ct.tf2ss(G_trimmed)

    # check stability
    evals = eigvals(Gss.A)
    if np.max(np.real(evals)) > 0:
        print("Algorithm only works for stable systems")
        return Gin

    Wc = solve_continuous_lyapunov(Gss.A, -Gss.B @ Gss.B.T)
    Wo = solve_continuous_lyapunov(Gss.A.T, -Gss.C.T @ Gss.C)
    Wc = (Wc + Wc.T)/2.0
    Wo = (Wo + Wo.T)/2.0
    L = scipy.linalg.cholesky(Wc, lower=True)
    hsv_original = hsv(Wc,Wo)

    if order is None: # following calc done using trimmed model 
        #order = Gss.A.shape[0] - 1
        order = pick_model_order_from_hsvs(hsv_original)
        print(f"Order not specified - selected {order:3d}")

    if order <= 0:
        print("System dimension not correct")
        return

    if method == 0:
        print("Using Method 0")
        Z = scipy.linalg.cholesky(Wo, lower=True)
        W, Sigma, Vh = svd(L.T @ Z)
        Sigma_sqrt_inv = np.linalg.inv(np.diag(np.sqrt(Sigma)))

        T = Sigma_sqrt_inv @ Vh @ Z.T
        Ti = L @ W @ Sigma_sqrt_inv

    elif method == 1:
        print("Using Method 1 - not recommended for high-order systems")

        Eigvals, K = eigh(L.T @ Wo @ L)
        idx = Eigvals.argsort()[::-1]
        Eigvals = Eigvals[idx]
        K = K[:, idx]

        Sigma = np.sqrt(Eigvals)
        Sigma_inv_sqrt = np.diag([1.0/xx for xx in np.sqrt(Sigma)])
        Sigma_sqrt = np.diag([xx for xx in np.sqrt(Sigma)])

        T  = Sigma_sqrt @ K.T @ np.linalg.inv(L)
        Ti = L @ K @ Sigma_inv_sqrt
    else:
        print("Method 0 or 1")
        return Gr

    Ab = T @ Gss.A @ Ti
    Bb = T @ Gss.B
    Cb = Gss.C @ Ti

    Arr = Ab[:order, :order]
    Brr = Bb[:order, :]
    Crr = Cb[:, :order]
    Drr = Gss.D
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
            Gr = ct.StateSpace(Arr, Brr, Crr, Drr)
        except Exception:
            print("Error in DCmatch step")
            pass

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
        print("\nOriginal model HSV: ",hsv_original)

    # add integrators back in state space
    if number_cut > 0:
        int_ss = ct.tf2ss(ct.tf([1], [1, 0]) ** number_cut)
        Gr = ct.series(Gr, int_ss) # statespace

    return Gr if is_ss else ct.ss2tf(Gr)

def pretty_row_print(X,msg="",sigfigs=None,decimals=3,complex_decimals=2,verbose=None):
    """
    Pretty print a row of real or complex numbers.

    Exactly one of sigfigs or decimals should be used.
    """

    if sigfigs is not None:
        decimals = None 

    # normalize scalar to 1 element array
    X = np.atleast_1d(X)

    def fmt_real(x):
        if sigfigs is not None:
            return f"{x:.{sigfigs}g}"
        else:
            return f"{x:.{decimals}f}"

    def fmt_complex(x):
        r = x.real
        i = x.imag

        if sigfigs is not None:
            r_str = f"{r:.{sigfigs}g}"
            i_mag_str = f"{abs(i):.{sigfigs}g}"
        else:
            r_str = f"{r:.{complex_decimals}f}"
            i_mag_str = f"{abs(i):.{complex_decimals}f}"

        # purely real
        if abs(i) < 1e-12:
            return r_str

        # purely imaginary
        if abs(r) < 1e-12:
            sign = "-" if i < 0 else ""
            return f"({sign}{i_mag_str}i)"

        # full complex
        sign = "-" if i < 0 else "+"
        return f"({r_str} {sign} {i_mag_str}i)"

    out = []
    for x in X:
        x = complex(x)
        if np.iscomplexobj(x) and abs(x.imag) > 0:
            out.append(fmt_complex(x))
        else:
            out.append(fmt_real(x.real))

    row = msg + " " + ", ".join(out)
    if verbose:
        return row
    else:
        display(Markdown(row))

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

    NG, DG = ct.tfdata(G)
    NG = np.atleast_1d(np.squeeze(NG))
    DG = np.atleast_1d(np.squeeze(DG))

    NC, DC = ct.tfdata(K)
    NC = np.atleast_1d(np.squeeze(NC))
    DC = np.atleast_1d(np.squeeze(DC))

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

    num, den = ct.tfdata(Gc)
    num = np.atleast_1d(np.squeeze(num))
    den = np.atleast_1d(np.squeeze(den))

    gain = float(num[0] / den[0]) if (len(num) and len(den)) else 0.0

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
def write_two_column_array(col1, col2, filename, sigfigs=4, title1='Residue',title2='Poles'):
    with open(filename, "w") as f:
        f.write("\\begin{array}{cc}\n")
        f.write(title1+" & "+title2+" \\\\\n")
        for a, b in zip(col1, col2):
            f.write(f"{a:.{sigfigs}f} & {b:.{sigfigs}f} \\\\\n")
        f.write("\\end{array}\n")


def write_latex_array(X, filename, msgs=None, cols=1, tol=1e-12, decimals=None, sigfigs=None):
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

    # accept sigfigs as alias for decimals
    if decimals is None and sigfigs is None:
        decimals = 2   # default
    elif decimals is None:
        decimals = sigfigs
    elif sigfigs is None:
        pass
    else:
        if decimals != sigfigs:
            raise ValueError("decimals and sigfigs must match if both are given")

    X = np.atleast_1d(X).astype(complex)

    used = np.zeros(len(X), dtype=bool)
    entries = []

    def fmt_real(x):
        return f"{x:.{decimals}f}"

    for i, z in enumerate(X):
        if used[i]:
            continue

        zr, zi = z.real, z.imag

        # try to find conjugate
        paired = False
        if abs(zi) > tol:
            for j in range(i + 1, len(X)):
                if used[j]:
                    continue
                zj = X[j]
                if (abs(zj.real - zr) < tol and
                    abs(zj.imag + zi) < tol):
                    # conjugate pair found
                    entries.append(
                        rf"({fmt_real(zr)} \pm {fmt_real(abs(zi))}i)"
                    )
                    used[i] = used[j] = True
                    paired = True
                    break

        if paired:
            continue

        # no conjugate pair
        if abs(zi) < tol:
            entries.append(fmt_real(zr))
        else:
            sign = "+" if zi >= 0 else "-"
            entries.append(
                rf"({fmt_real(zr)} {sign} {fmt_real(abs(zi))}i)"
            )
        used[i] = True

    # split into rows
    rows = [
        entries[i:i+cols]
        for i in range(0, len(entries), cols)
    ]

    with open(filename, "w") as f:
        f.write("\\begin{array}{%s}\n" % ("c" * cols))
        if msgs:
            f.write(msgs + "\n")
        for r in rows:
            f.write("  " + " & ".join(r) + " \\\\\n")
        f.write("\\end{array}\n")

def show_tf_latex(P, label=None, sigfigs=2, show=None, factor=False,
                  name=None, time_constant=False):
    ''' 
    P: system
    label
    show
    factor
    time_constant: if True, normalize first order real factors to (s/a + 1)
    '''

    is_discrete = P.dt is not None and P.dt > 0
    var = "z" if is_discrete else "s"
    
    if label is None:
        label = f"G({var})"
    if name is not None:
        label = name

    num, den = ct.tfdata(P)
    num = np.atleast_1d(np.squeeze(num))
    den = np.atleast_1d(np.squeeze(den))

    if factor:
        Kn, rnum, qnum = factor_poly_real(num)
        Kd, rden, qden = factor_poly_real(den)

        # cancel common real roots (returns deterministic sorted lists now)
        rnum_c, rden_c = cancel_common_real_roots(rnum, rden, tol=1e-6)

        # ensure deterministic ordering (in case upstream callers pass unsorted lists)
        rnum_c = sorted(rnum_c, key=lambda r: (abs(r), r))
        rden_c = sorted(rden_c, key=lambda r: (abs(r), r))

        # quadratics: sort by C (which is a^2 + b^2) then B for stability
        qnum = sorted(qnum, key=lambda bc: (bc[1], bc[0]))
        qden = sorted(qden, key=lambda bc: (bc[1], bc[0]))

        # apply time-constant normalization AFTER sorting the physical roots so ordering
        # is done on the actual root locations (more intuitive)
        if time_constant:
            def normalize_real_roots(rlist):
                new_roots = []
                gain_scale = 1.0
                for r in rlist:
                    if np.isreal(r):
                        r = float(np.real(r))
                        a = -r
                        if a != 0:
                            gain_scale *= a
                            new_roots.append(-a)  # store as root of (s/a + 1)
                        else:
                            new_roots.append(r)
                    else:
                        new_roots.append(r)
                return np.array(new_roots), gain_scale

            rnum_c, scale_num = normalize_real_roots(rnum_c)
            rden_c, scale_den = normalize_real_roots(rden_c)

            Kn *= scale_num
            Kd *= scale_den

        # build latex bodies with the now-ordered lists
        num_body = factors_to_latex(rnum_c, qnum, var, sigfigs,
                                    time_constant=time_constant)
        den_body = factors_to_latex(rden_c, qden, var, sigfigs,
                                    time_constant=time_constant)

        frac = build_frac_latex_gain_in_numer(Kn, num_body, Kd, den_body, sigfigs)

    else:
        num_tex = _poly_to_latex(num, sigfigs=sigfigs, var=var, discrete=is_discrete)
        den_tex = _poly_to_latex(den, sigfigs=sigfigs, var=var, discrete=is_discrete)
        frac = rf"\dfrac{{{num_tex}}}{{{den_tex}}}"

    latex_str = rf"${label} = {frac}$"

    if show:
        display(Math(latex_str))
        return None

    return latex_str

if 0:
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

def tf_to_latex(G, sigfigs=2, factor=False, time_constant=False):
    """
    Backward-compatible wrapper for show_tf_latex.
    Returns LaTeX string (without surrounding $).
    """
    latex_str = show_tf_latex(G,label=None,sigfigs=sigfigs,
        show=False,factor=factor,time_constant=time_constant)
    # remove outer $...$ added by show_tf_latex
    if latex_str.startswith("$") and latex_str.endswith("$"):
        latex_str = latex_str[1:-1]
    return latex_str

if 0:
    def tf_to_latex(G):
        s = sp.Symbol('s')
        
        # Get numerator and denominator coefficients
        num, den = ct.tfdata(G)
        num = np.atleast_1d(np.squeeze(num))
        den = np.atleast_1d(np.squeeze(den))

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
        return rf"\displaystyle {fmt(K, sigfigs)}\,\dfrac{{1}}{{{den_body}}}"

    if abs(K - 1) < tol:
        return rf"\displaystyle \dfrac{{{num_body}}}{{{den_body}}}"

    return rf"\displaystyle {fmt(K)}\,\dfrac{{{num_body}}}{{{den_body}}}"

def residue_tf(G, time_constant=False, tol=1e-12):
    """
    Partial fraction expansion of transfer function G.

    Standard form:
        r, a, k  corresponds to  r / (s + a)

    Time constant form:
        r_tc, a, k  corresponds to  r_tc / (s/a + 1)

        where
            a = -p
            r_tc = r / a

        For a stable pole p < 0, a > 0.
        Integrators (p ≈ 0) are returned as a = 0 and r/s.
    """
    num = np.squeeze(G.num)
    den = np.squeeze(G.den)
    r, p, k = residue(num, den)

    if not time_constant:
        return r, -p, k

    r_tc = []
    a_vals = []
    for ri, pi in zip(r, p):
        # integrator
        if abs(pi) < tol:
            r_tc.append(ri)
            a_vals.append(0.0)
            continue

        a = -pi            # define break frequency
        r_new = ri / a     # rescale residue

        r_tc.append(r_new)
        a_vals.append(a)

    return np.array(r_tc), np.array(a_vals), k

def factors_to_latex(real_roots, quads, var="s",sigfigs=4, tol=1e-8,time_constant=False):

    parts = []

    # ---- handle real roots ----
    real_roots = np.asarray(real_roots, dtype=float)

    # count zero roots separately (integrators)
    zero_mask = np.abs(real_roots) < tol
    n_zero = np.sum(zero_mask)

    # emit integrator factor
    if n_zero == 1:
        parts.append(var)
    elif n_zero > 1:
        parts.append(f"{var}^{n_zero}")

    # nonzero real roots
    nz_roots = real_roots[~zero_mask]

    if len(nz_roots) > 0:
        # round first to stabilize duplicates
        nz_roots = np.round(nz_roots, sigfigs)

        # cluster WITHOUT reordering
        clustered = []
        for r in nz_roots:
            if not clustered:
                clustered.append([r, 1])
            else:
                if abs(r - clustered[-1][0]) <= tol:
                    clustered[-1][1] += 1
                else:
                    clustered.append([r, 1])

        for r, mult in clustered:
            a = -r  # since factor is (s - r) = (s + a)

            if time_constant and abs(a) > tol:
                a_fmt = fmt(abs(a), sigfigs)

                if a > 0:
                    factor = rf"\left(\frac{{{var}}}{{{a_fmt}}}+1\right)"
                else:
                    factor = rf"\left(\frac{{{var}}}{{{a_fmt}}}-1\right)"

            else:
                if a > 0:
                    factor = f"({var}+{fmt(a, sigfigs)})"
                else:
                    factor = f"({var}-{fmt(abs(a), sigfigs)})"

            # attach exponent if repeated
            if mult > 1:
                factor += f"^{mult}"

            parts.append(factor)

    # ---- handle quadratic factors ----
    for B, C in quads:
        quad = f"{var}^2"

        # B term
        if abs(B) > tol:
            signB = "-" if B < 0 else "+"
            quad += f"{signB}{fmt(abs(B), sigfigs)}{var}"

        # C term
        if abs(C) > tol:
            signC = "-" if C < 0 else "+"
            quad += f"{signC}{fmt(abs(C), sigfigs)}"

        parts.append(f"({quad})")

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
            return rf"\displaystyle \dfrac{{1}}{{{db}}}"
        elif abs(K_total + 1) < Tol:
            return rf"\displaystyle -\dfrac{{1}}{{{db}}}"
        else:
            return rf"\displaystyle {K_tex}\,\dfrac{{1}}{{{db}}}"

    # both present
    if abs(K_total - 1) < Tol:
        return rf"\displaystyle \dfrac{{{nb}}}{{{db}}}"
    elif abs(K_total + 1) < Tol:
        return rf"\displaystyle -\dfrac{{{nb}}}{{{db}}}"
    else:
        return rf"\displaystyle {K_tex}\,\dfrac{{{nb}}}{{{db}}}"


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

    Mathtext-safe version:
      - no \\displaystyle
      - no \\dfrac
      - suitable for Matplotlib and notebooks
    """

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
        return rf"{Ktex}"

    # numerator only
    if db is None:
        if abs(K - 1) < Tol:
            return rf"{nb}"
        elif abs(K + 1) < Tol:
            return rf"-{nb}"
        else:
            return rf"{Ktex}\,{nb}"

    # denominator only
    if nb is None:
        if abs(K - 1) < Tol:
            return rf"\dfrac{{1}}{{{db}}}"
        elif abs(K + 1) < Tol:
            return rf"-\dfrac{{1}}{{{db}}}"
        else:
            return rf"\dfrac{{{Ktex}}}{{{db}}}"

    # full fraction
    if abs(K - 1) < Tol:
        return rf"\dfrac{{{nb}}}{{{db}}}"
    elif abs(K + 1) < Tol:
        return rf"-\dfrac{{{nb}}}{{{db}}}"
    else:
        return rf"\dfrac{{{Ktex}\,{nb}}}{{{db}}}"

def group_real_roots(real_roots, tol=1e-6):
    groups = []
    for r in real_roots:
        for g in groups:
            if abs(r - g[0]) < tol:
                g.append(r)
                break
        else:
            groups.append([r])
    return groups

def poly_factors_to_latex(K, real_roots, quads, sigfigs=4):
    terms = []

    # real roots
    for g in group_real_roots(real_roots):
        r = g[0]
        mult = len(g)

        # ---- HANDLE ZERO ROOT CLEANLY ----
        if abs(r) < 1e-12:
            term = "s"
            if mult > 1:
                term += f"^{mult}"
            terms.append(term)
            continue
        # -----------------------------------

        a = -r
        term = f"(s {'+' if a >= 0 else '-'} {abs(a):.{sigfigs}g})"
        if mult > 1:
            term += f"^{mult}"
        terms.append(term)

    # quadratic factors
    for B, C in quads:
        terms.append(
            f"(s^2 + {B:.{sigfigs}g}s + {C:.{sigfigs}g})"
        )

    # no factors → return empty string, not "1"
    body = "".join(terms)

    # only include K if it is not 1
    if abs(K - 1.0) > 1e-12:
        return f"{K:.{sigfigs}g}" + body

    return body

def _poly_to_latex(coefs, sigfigs=4, var="s", discrete=False, Tol = 1e-12):
    terms = []

    coefs = np.atleast_1d(coefs).astype(float)

    # Trim leading near-zero coefficients
    while len(coefs) > 1 and abs(coefs[0]) < Tol:
        coefs = coefs[1:]

    n = len(coefs)

    for i, val in enumerate(coefs):

        if abs(val) < Tol:
            continue

        # sign and magnitude
        sign = "-" if val < 0 else "+"
        mag = abs(val)

        # format once to significant figures
        coeff_str = f"{mag:.{sigfigs}f}"
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
                term_body = rf"{'' if abs(mag-1.0)<Tol else coeff_str}{var}^{degree}"
            elif degree == 1:
                term_body = rf"{'' if abs(mag-1.0)<Tol else coeff_str}{var}"
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

def factor_poly_real(coeffs, tol=1e-6):
    """
    coeffs: highest to lowest
    returns:
        K          : leading coefficient
        real_roots : list of real roots (sorted by increasing |root|)
        quads      : list of (B, C) for s^2 + B s + C,
                     sorted by increasing |root|
    """
    coeffs = np.atleast_1d(np.asarray(coeffs, dtype=float))

    if coeffs.size == 1:
        return coeffs[0], [], []

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

        for j in range(i + 1, len(roots)):
            if not used[j] and abs(roots[j] - np.conj(r)) <= imag_tol:
                used[i] = used[j] = True
                a = r.real
                b = abs(r.imag)
                quads.append((-2*a, a*a + b*b))
                break

    # -------- NEW: cluster real roots --------
    real_roots.sort()
    clustered = []

    for r in real_roots:
        if not clustered:
            clustered.append([r, 1])
        else:
            if abs(r - clustered[-1][0]) <= tol:
                clustered[-1][1] += 1
            else:
                clustered.append([r, 1])

    # convert back to list with multiplicity
    # convert back to list with multiplicity
    real_roots = []
    for r, mult in clustered:
        real_roots.extend([r]*mult)

    # final deterministic ordering
    real_roots = sorted(real_roots, key=lambda r: (abs(r), r))
    quads = sorted(quads, key=lambda q: (q[1], q[0]))

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

def write_tf_latex(G, filename, label, sigfigs=4,
                   factor=False, inline=False,
                   name=None, time_constant=False,show=False):

    latex_str = show_tf_latex(G,label=label,sigfigs=sigfigs,show=False,
        factor=factor,name=name,time_constant=time_constant)

    # remove outer $...$ from show_tf_latex
    if latex_str.startswith("$") and latex_str.endswith("$"):
        latex_str = latex_str[1:-1]

    with open(filename, "w") as f:
        if inline:
            f.write("$\n")
            f.write(latex_str + "\n")
            f.write("$\n")
        else:
            f.write("\\[\n")
            f.write(latex_str + "\n")
            f.write("\\]\n")

def normalize_tf(G):
    '''factor out non-unity gain for leading coefficient of the denominator'''
    if isinstance(G, ct.StateSpace):
        G = ct.ss2tf(G)

    num, den = ct.tfdata(G)
    num = np.atleast_1d(np.squeeze(num))
    den = np.atleast_1d(np.squeeze(den))

    if den[0] != 0:
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

def plot_spec_region(ax, zeta, wn, wd, color='m', highlight_color='r', linestyle='--'):
    """
    Draws the damping/angle spec lines used in your plots, but using the
    current axis limits (no magic 20). Works for both full and zoomed axes.
    """
    # angle from geometry
    th = np.arctan2(wd, (zeta*wn))

    # current axis bounds
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # vertical damping line at real = -zeta*wn, extend it to the full vertical axis
    x_vert = -zeta * wn
    ax.plot([x_vert, x_vert], [ymin, ymax], color=color, linestyle=linestyle)

    # Angled lines: these were originally y = -x * tan(th) (so they pass through origin).
    # Build a line from left axis limit to the damping vertical line.
    x_line = np.array([xmin, x_vert])
    y_line = -x_line * np.tan(th)   # y = -x * tan(th)
    ax.plot(x_line, y_line, color=color, linestyle=linestyle)
    ax.plot(x_line, -y_line, color=color, linestyle=linestyle)  # symmetric lower branch

    # Short connectors from origin to damping line (0 -> -zeta*wn)
    x_conn = np.array([0.0, x_vert])
    y_conn = -x_conn * np.tan(th)
    ax.plot(x_conn, y_conn, color=color, linestyle=linestyle)
    ax.plot(x_conn, -y_conn, color=color, linestyle=linestyle)

    # Now draw the solid highlight (same geometry as above but solid and only the local segment)
    # Use the mph segment limited to the local spec (i.e., -wd..wd around the damping vertical)
    ax.plot([x_vert, x_vert], [-wd, wd], color=highlight_color, linestyle='-')

    # solid angled highlight between xmin and x_vert, but clip to visible y-range so we don't overdraw
    y_line_high = -x_line * np.tan(th)
    ax.plot(x_line, y_line_high, color=highlight_color, linestyle='-')
    ax.plot(x_line, -y_line_high, color=highlight_color, linestyle='-')
