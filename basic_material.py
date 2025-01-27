from platform import python_version
print("Running Python:",python_version())

import shutil, sys, os.path, math, time, subprocess, random, importlib.util

import numpy as np
from numpy import logspace, linspace
float_formatter = "{:.4f}".format
np.set_printoptions(formatter={'float': '{: 8.3f}'.format})

import matplotlib
import matplotlib.cm as cm
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.pyplot import figure, savefig
from matplotlib import gridspec
from matplotlib import rcParams
rcParams["font.serif"] = "cmr14"
rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [8, 5.0]
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.autolayout"] = True

import sympy as sym
from sympy import lambdify, oo, Symbol, integrate, Heaviside, plot, Piecewise
from sympy import exp, plot, sin, cos, printing, init_printing, simplify
from sympy.testing.pytest import ignore_warnings
print("Running Sympy:",sym.__version__)
#init_printing(use_unicode=True)

from scipy import signal
from scipy.fft import fft, fftfreq, fftshift, ifft

try:
    import IPython.display as ipd
except Exception as e2:
    print(e2)
    subprocess.Popen('python3 -m pip install IPython', shell=True)    
    import IPython.display as ipd

from IPython.display import display, Markdown
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import warnings
warnings.filterwarnings("ignore")

r2d = 180/np.pi
rps2hz = 1/(2*np.pi)

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# where I tend to put data files
source = "/Users/jonathanhow/Dropbox (MIT)/Classes/16.06/Spr_2025/Source/data/"


####################################################################
####################################################################
# my step function that can be used in plotting
def U(t):
    u = np.zeros.like(t)
    u[t >= 0] = 1
    return u


####################################################################
####################################################################
# makes a nice grid of various line widths/shades
def nicegrid(ax, hh = -1): # hh < 0 is used to swithc off behavior that is not useful
    try: #if np.size(ax) > 1
        for ii in np.arange(len(ax)):
            jgrid(ax[ii],hh)
    except:
        jgrid(ax,hh)
            

####################################################################
####################################################################
# grid helper
def jgrid(ax,hh = 9):
    ax.grid(True, which='major', color='#666666', linestyle=':')
    ax.grid(True, which='minor', color='#999999', linestyle=':', alpha=0.2)
    try:
        ax.axhline(y=0, color='k', linestyle='-',lw=1)
        ax.axvline(x=0, color='k', linestyle='-',lw=1)
    except:
        ax.axhline(y=1, color='k', linestyle='--',lw=1)
    ax.minorticks_on()
    if hh > 0:
        ax.xaxis.set_major_locator(ticker.LinearLocator(hh))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    

####################################################################
####################################################################
# caption below figure, sometimes useful
def caption(txt,fig, xloc=0.5, yloc=-0.05):
    fig.text(xloc, yloc, txt, ha='center',size=MEDIUM_SIZE,color='blue')


####################################################################
####################################################################
if not os.path.isdir("./data/"):
    os.mkdir("./data")

if not os.path.isdir("./figs/"):
    os.mkdir("./figs")


####################################################################
####################################################################
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False


####################################################################
####################################################################
try: 
    from simple_colors import *
    colors = ['b','r','m','g','k','Brown','DarkBlue','Tomato','Violet', 'Tan','Salmon','Pink',
    'SaddleBrown', 'SpringGreen', 'RosyBrown','Silver',]
except:
    colors = ['b','r','m','g','k']


####################################################################
####################################################################
if importlib.util.find_spec('control') is None:
    print("Installing Control Package")
    os.system(f'python -m pip install control')
else:
    print("Control Package Found")

if importlib.util.find_spec('slycot') is None:
    slycot_available = False
    # uncomment to install slycot package
    #os.system(f'pip install slycot')
else:
    slycot_available = True

####################################################################
####################################################################
if __name__ == "__main__":
    pass
else:
    print("This is a library of basic functions for 16.06")