import numpy as np 
import cmath 
from sympy import Symbol, atan, pi, tan, sqrt, solve, log, symbols
from numpy.polynomial import Polynomial
from numpy import inf

import control as ct
ct.set_defaults('nyquist',max_curve_magnitude = 100)
import control
import control.matlab
from control.matlab import feedback, tf

import matplotlib.pyplot as plt
from scipy.optimize import minimize

r2d = 180/np.pi
tpi = 2*np.pi

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

####################################################################
####################################################################
# add break-in gains to improve plot
def Root_Locus_gains(L, Krange = None, Tol = 1e-3, standard_locus = True, Tol_max = 1e3):
    ''' 
    Augment the RL gains to include the break-in/break-out pts
    Thus leading to a better plot.

    In:     L(s)    - system for RL assumes TF form
            Krange  - set of K values
            standard_locus - True for RL with positive K values, False for negative K values
            Tol_max - maximum value of K to consider
            Tol - tolerance for real part of s (nearness to real axis)

    Out:    Krange  - augmented set of K values

    '''
    if Krange is None:
        Krange = (2*standard_locus-1)*np.logspace(-3, 3, num=2000)
    Krange = np.sort(np.append(Krange,0)) # add zero
    
    try:
        Num = L.num[0][0]
        Den = L.den[0][0]
        dNds = np.polyder(Num)
        if dNds.size == 0:
            dNds = np.array([0])
        dDds = np.polyder(Den)
        if dDds.size == 0:
            dDds = np.array([0])
        part1 = np.convolve(dNds, Den)
        part2 = np.convolve(Num, dDds)
        # make sure same size so we can add them
        max_len = max(len(part1), len(part2))
        part1 = np.pad(part1, (max_len - len(part1), 0), 'constant')
        part2 = np.pad(part2, (max_len - len(part2), 0), 'constant')
        pdr = np.roots(part1 - part2) # poles of dL/ds

        Kkeep = [-1/np.real(L(x)) for x in pdr if abs(x.imag) < Tol] # k = -1/L(s) if s in pdr is real
        if standard_locus: # only look at the relevant sign K values depending on which RL is being drawn
            Kkeep = [x for x in Kkeep if ((x >= 0) and (x < Tol_max))] 
        else:
            Kkeep = [x for x in Kkeep if ((x <= 0) and (x > -Tol_max))]
        
        if len(Kkeep) > 0:
            Krange = np.sort(np.append(Krange,Kkeep))
            npoles = len(L.den[0][0])
            nzeros = len(L.num[0][0])
            n_add = int(npoles - nzeros) # add 0's to match den length
            L_num_add = np.pad(L.num[0][0], (n_add,0), 'constant', constant_values=(0))
            for kk in Kkeep:
                phi_temp = L.den[0][0] + kk*L_num_add # clp denominator for that K
                scl = np.roots(phi_temp)
                real_poles = [np.round(x.real,3) for x in scl if abs(x.imag) < Tol]
                double_real_poles = set([x for x in real_poles if real_poles.count(x) > 1])
                print("\nFound breakin/out at K = {:4.3f}".format(kk))
                print("At possible locations s = "+', '.join('{:4.3f}'.format(x.real) for x in double_real_poles))
        else:
            double_real_poles = []
    except:
        print("Gain augmentation failed")
    return Krange 
    
####################################################################
####################################################################
def RL_COM(L,standard_locus = True):
    ''' 
    Find the CoM of a RL for L(s)

    in:     L(s)    - system for RL

    out:    CoM     - Location of CoM    
            Ang     - angle of asymptotes in degrees

    if # poles = # zeros then avoids division by zero and returns None
    '''
    np = len(L.poles()) 
    nz = len(L.zeros())
    
    if np <= nz: # no asymptotes or improper system
        CoM = None
        Ang = None
        return CoM, Ang
    
    if np == (nz + 1): # one asymptote
        CoM = None
        if standard_locus:
            Ang = 180.0
        else:
            Ang = 0.0
        return CoM, Ang
    
    # here np >= nz + 2    
    # recall that we need 2 more poles than zeros for CoM of asymptotes to exist
    CoM = (sum([x for x in L.poles()]) - sum([x for x in L.zeros()]))/(np - nz)
    if standard_locus:
        Ang = 180.0/(np - nz) % 360.0
    else:
        Ang = 360.0/(np - nz) % 360.0
    return CoM, Ang

####################################################################
####################################################################
def Root_Locus_design_cancel(G, s_target = complex(-1,2), s_cancel = -1, verbose = False):
    '''
    RL Lead design of Gc by placing/canceling pole at s_cancel to ensure that CLP are at s_target
    '''
    phi_fromG = sum([cmath.phase(x) for x in (s_target - G.zeros())])*r2d - \
                sum([cmath.phase(x) for x in (s_target - G.poles())])*r2d

    Gczeros = np.array([np.real(s_cancel)]) # place *real* zero
    phi_from_Gc_zero = sum([cmath.phase(x) for x in (s_target - Gczeros)])*r2d
    phi_required = (180 + phi_fromG + phi_from_Gc_zero)%360
    
    # now solve the phase condition equation for the comp pole location
    P = s_target.imag/np.tan(phi_required/r2d) - s_target.real
    Gc = tf((1,-Gczeros[0]), (1,P))
    Gain = -1/np.real(G(s_target) * Gc(s_target))
    Gc *= Gain
    Gc_zeros = Gc.zeros()[0].real
    Gc_poles = Gc.poles()[0].real
    L = G*Gc    
    Gcl = feedback(L,1)

    if verbose:
        print(f"{phi_fromG = :4.2f}")
        print(f"{phi_required = :4.2f}")
        print(f"{phi_from_Gc_zero = :4.2f}")
        print(f"{Gc_zeros = :4.2f}")
        print(f"{Gc_poles = :4.2f}")
        print(f"{Gain = :4.2f}")

    return Gc, Gcl.poles()

####################################################################
####################################################################
def Root_Locus_design_ratio(G, s_target = complex(-1,2), gamma = 10, z0 = None, idx = None, verbose = False):
    '''
    RL Lead design of Gc by to put CLP at s_target using a Gc.p/Gc.z = gamma
    '''
    z = Symbol('z')
    if verbose:
        phi_fromG = sum([cmath.phase(x) for x in (s_target - G.zeros())]) - \
                    sum([cmath.phase(x) for x in (s_target - G.poles())])
        print(f"{phi_fromG*r2d = :4.2f}")
        phi_required = (np.pi - phi_fromG)%(2*np.pi)
        print(f"{phi_required*r2d = :4.2f}")

    def func(z, gam, G, s_0):
        Gc = tf((1, float(z)), (1, float(gam*z)))  # comp with zero at z and pole at p
        L = Gc*G
        phi_fromL = (sum([cmath.phase(x) for x in (s_0 - L.zeros())]) * 180 / np.pi - \
                    sum([cmath.phase(x) for x in (s_0 - L.poles())]) * 180 / np.pi) % 360
        return (phi_fromL - 180) % 360    

    if z0 is None:
        z0 = -s_target.real/2  # default initial guess for zero location
        
    resPID = minimize(func, x0=z0, args=(gamma,G,s_target,), tol=1e-3, method='Nelder-Mead', options={'disp': verbose, 'maxiter': 1000})
    if not resPID.success:
        print("Optimization failed")
    else:
        if verbose:
            print(f"Optimization success: {resPID.success}")
            pretty_row_print(resPID.x, "Optimized z: ")

    if idx is None:
        Gczeros = resPID.x[0]  # real zero location
    else:
        Gczeros = resPID.x[idx]
        
    Gc = tf((1, float(Gczeros)), (1,float(gamma*Gczeros)))
    Gain = -1/np.real(G(s_target) * Gc(s_target))
    Gc *= Gain

    if verbose:
        print(f"Optimized Gc zero location: {Gczeros = :4.2f}")
        print(f"Optimized Gc pole location: {gamma*Gczeros = :4.2f}")
        print(f"{Gain = :4.2f}")

    L = G*Gc
    Gcl = feedback(L)
    Gcl.poles()

    return Gc, Gcl.poles()
  
####################################################################
####################################################################
def Root_Locus_design_PD(G, s_target = complex(-1,2),verbose=False):
    '''
    RL PD design of Gc by to put CLP at s_target 
    '''
    phi_fromG = sum([cmath.phase(x) for x in (s_target - G.zeros())]) - \
                sum([cmath.phase(x) for x in (s_target - G.poles())])
    print(f"{phi_fromG*r2d = :4.2f}")
    phi_required = (np.pi - phi_fromG)%(2*np.pi)
    print(f"{phi_required*r2d = :4.2f}")

    # now solve the phase condition equation for the comp pole location
    Z = s_target.imag/np.tan(phi_required) - s_target.real
    Gc = tf((1,Z),1)
    Gain = -1/np.real(G(s_target) * Gc(s_target))
    Gc *= Gain
    L = G*Gc    
    Gcl = feedback(L,1)

    if verbose:
        print(f"Gc Zero {Z = :4.2f}")
        print(f"{Gain = :4.2f}")

    return Gc, Gcl.poles()

####################################################################
####################################################################
class Step_info:
    # init method or constructor
    def __init__(self,t,y, method = 0, t0 = 0, SettlingTimeLimits = [0.02], RiseTimeLimits = [0.1,0.9]):
        self.t = t
        self.y = y
        self.Yss = y[-1] # approx steady state value
        self.SettlingTimeLimits = SettlingTimeLimits
        sgnYss = np.sign(self.Yss.real)

        tr_lower_index = np.where(sgnYss * (self.y - RiseTimeLimits[0] * self.Yss) >= 0)[0][0]
        tr_upper_index = np.where(sgnYss * (self.y - RiseTimeLimits[1] * self.Yss) >= 0)[0][0]
        self.Tr = self.t[tr_upper_index] - self.t[tr_lower_index]
        self.Tr_values = [self.t[tr_lower_index] - t0,self.t[tr_upper_index] - t0]

        #Find the time that has settled close to the steady state value
        settled = np.where(np.abs(self.y/self.Yss-1) >= SettlingTimeLimits)[0][-1]+1
        if settled < len(self.t):
            self.Ts = self.t[settled] - t0
        else:
            self.Ts = 0.  # avoids weird plot
        
        # Peak overshoot
        self.Mp = (self.y.max()/self.Yss-1)
        self.Tp = t[int(np.median(np.argwhere(self.y == self.y.max())))] - t0
        if method == 0: # which methods used to estimate zeta and wn from the step results
            print("Using Tp")
            self.zeta = 1/np.sqrt( 1 + (np.pi/np.log(self.Mp))**2 ) 
            self.wn = np.pi/self.Tp/np.sqrt(1-self.zeta**2)
        else:
            print("Using Ts")
            q = self.Tp/np.pi/self.Ts
            if self.SettlingTimeLimits[0] == 0.01:
                q *= 4.6 # 1% rule
            else:
                q *= 4 # 2 % rule
            self.zeta = q / np.sqrt( 1 + q**2 )
            self.wn = 4/self.Ts/self.zeta
    
    def printout(self, raw = False):
        print("omega_n: \t%4.3f"%(self.wn))
        print("zeta   : \t%4.3f"%(self.zeta))
        print("Tr     : \t%4.2fs"%(self.Tr))
        print("Ts     : \t%4.2fs"%(self.Ts))
        print("Mp     : \t%4.2f"%(self.Mp))
        print("Tp     : \t%4.2fs"%(self.Tp))
        print("Yss    : \t%4.2f"%(self.Yss))
       
    def nice_plot(self,ax, Tmax = None, Ymax = None):
        if Ymax is None:
            ylim=(np.floor(np.min(self.y)),np.ceil(10.*np.max(self.y))/10.0)
            Ymax = np.max(ylim) # needed for plot scaling
        if Tmax is None:
            Tmax = np.max(self.t)

        try:
            print(f"Using {self.SettlingTimeLimits[0] = :4.2f}")
            self.SettlingTimeLimits = self.SettlingTimeLimits[0]
        except:
            print(f"Using {self.SettlingTimeLimits = :4.2f}")
            
        # the response
        ax.plot(self.t,self.y,'b')
        # vertical lines at Tr, Tp, Ts
        ax.axvline(x = self.Tr_values[0],ymax=0.1*self.Yss/Ymax,c='r',ls='dashed')
        ax.axvline(x = self.Tr_values[1],ymax=0.9*self.Yss/Ymax,c='r',ls='dashed')
        ax.axvline(x = self.Ts,ymax=self.Yss/Ymax,c='grey',ls='dashed')
        ax.axvline(ymax = self.Yss*(1 + self.Mp)/Ymax, x=self.Tp, c='m',ls='dashed',lw=2)
        # horizontal lines at Yss, Mp, SettlingTimeLimits
        ax.axhline(y = (1+self.SettlingTimeLimits)*self.Yss,xmin=self.Ts/Tmax,c='grey',ls='dashed',lw=1)
        ax.axhline(y = (1-self.SettlingTimeLimits)*self.Yss,xmin=self.Ts/Tmax,c='grey',ls='dashed',lw=1)
        ax.plot((0, self.Tp), (self.Yss*(1 + self.Mp), self.Yss*(1 + self.Mp)), c='green',ls='dashed',lw=2)
        # add text to the plot
        ax.text(self.Tr/2, 0.25*self.Yss, "Tr = {0:.2f}".format(self.Tr), fontsize=SMALL_SIZE)
        ax.text(self.Tp, 0.75*self.Yss, "Tp = {0:.2f}".format(self.Tp), fontsize=SMALL_SIZE)
        ax.text(self.Ts, 0.5*self.Yss, "Ts = {0:.2f}".format(self.Ts), fontsize=SMALL_SIZE)
        ax.text(self.Tp*1.1, self.Yss*(1 + self.Mp), "Mp = {0:.2f}".format(self.Mp), fontsize=SMALL_SIZE)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('Response')
        ax.set_title('Step Response')
        ax.set_ylim(0,Ymax)
        ax.set_xlim(0,Tmax)

####################################################################
####################################################################
def lead_design(G, wc_des = 1, PM = 45):
    j = complex(0,1)
    Gf = G(j*wc_des)
    phi_G = cmath.phase(Gf)*r2d
    phi_m = (PM - (180 + phi_G))/r2d # robust?
    zdp = (1 - np.sin(phi_m))/(1 + np.sin(phi_m))
    z = wc_des*np.sqrt(zdp)
    p = z/zdp
    Gc_lead = tf([1, z],[1, p]) 
    L = G*Gc_lead
    k_c = 1/np.abs(L(j*wc_des))
    Gc_lead *= k_c
    return Gc_lead

####################################################################
####################################################################
def lag_design(gain_inc = 10, gamma = 10, wc = 1):
    '''
    gain_inc: goal for adding the lag
    gamma = 10: heuristic design approach
    wc: design goal
    '''
    zl = wc/gamma 
    pl = zl/gain_inc
    Gc_lag = tf([1, zl],[1, pl]) # lag comp
    return Gc_lag 
                
####################################################################
####################################################################
# how many free integrators are there?
def find_system_type(L):
    return len(L.den[0][0]) - len(np.trim_zeros(L.den[0][0], 'b'))

####################################################################
####################################################################
def find_Kp(L):
    L_type = find_system_type(L)
    if L_type == 0:
        return np.real((L.num[0][0][-1]/L.den[0][0][-1]))
    else:
        return None

####################################################################
####################################################################
def find_Kv(L):
    L_type = find_system_type(L)
    if L_type == 0:
        return 0
    elif L_type == 1:
        return np.real(L.num[0][0][-1]/L.den[0][0][-2])
    else:
        return None

####################################################################
####################################################################
def find_Ka(L):
    L_type = find_system_type(L)
    if L_type < 2:
        return 0
    elif L_type == 2:
        return np.real(L.num[0][0][-1]/L.den[0][0][-3])
    else:
        return None

####################################################################
####################################################################
# find frequency of gain crossover
def find_wc(omega, G, mag = 1):
    '''
    find freq when the system mag = mag 
    '''
    Gf = G(1j*omega)  # complex freq response
    idx = np.argmin(np.abs(mag - np.abs(Gf)))  # find the index where |G(jw)| is closest to mag
    return omega[idx], idx  # return the frequency and index

# find the gain at phase crossover (-pi)
def find_wpi(omega, G, phi = np.pi):
    '''
    find freq when system phase = pi 
    '''
    Gf = G(1j*omega)  # complex freq response
    idx = np.argmin(np.abs(phi - np.angle(Gf) * r2d))
    return omega[idx], idx

####################################################################
####################################################################
# phase shift to recenter around 0
def pshift(Gp):
    ''' 
    shift phase to be between -pi and pi
    '''
    while (np.max(Gp) < -np.pi):
        Gp += 2*np.pi
    while (np.min(Gp) > np.pi):
        Gp -= 2*np.pi
    return Gp
    
####################################################################
####################################################################
def caption(txt,fig, xloc=0.5, yloc=-0.05):
    fig.text(xloc, yloc, txt, ha='center',size=MEDIUM_SIZE,color='blue')

####################################################################
####################################################################
def my_pzmap(G,ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.real(G.poles()),np.imag(G.poles()),'bx',ms=6,markerfacecolor=None)
    ax.plot(np.real(G.zeros()),np.imag(G.zeros()),'o',ms=6,markeredgewidth=2, markeredgecolor='r',markerfacecolor='r')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Pole-Zero Map')
    ax.grid(True)
    return ax

def color_rl(ax):
    for kk in range(0, len(ax.lines)):
        #print(ax.lines[kk].get_marker())
        if ax.lines[kk].get_linestyle() == '-':
            ax.lines[kk].set_linewidth(1.5)
            ax.lines[kk].set_color('blue')
        if ax.lines[kk].get_marker() == 'x':
            ax.lines[kk].set_markersize(8)
            ax.lines[kk].set_color('blue')
        if ax.lines[kk].get_marker() == 'o':
            ax.lines[kk].set_markersize(8)
            ax.lines[kk].set_markerfacecolor('r')
            ax.lines[kk].set_markeredgecolor('r')
        if ax.lines[kk].get_marker() == 'd':
            ax.lines[kk].set_markersize(8)
            ax.lines[kk].set_markerfacecolor('g')
            ax.lines[kk].set_markeredgecolor('g')
            
####################################################################
####################################################################
def Read_data(file_name,comments=['#','F'],cols=[0]):
    '''
    Full file_name
    comments=['#','F'] for AD2 data
    cols=[0]
    '''
    return np.loadtxt(file_name,comments=comments,delimiter=',',usecols=cols)

####################################################################
####################################################################
# remove weird scaling artifacts
def near_zero(P, Tol = 1e-12):
    P.num[0][0] = [x if abs(x) > Tol else 0.0 for x in P.num[0][0]]
    P.den[0][0] = [x if abs(x) > Tol else 0.0 for x in P.den[0][0]]
    return tf(P.num,P.den)

####################################################################
####################################################################
def log_interp(zz, xx, yy):
    logz = np.log10(zz)
    logx = np.log10(xx)
    logy = np.log10(yy)
    return np.power(10.0, np.interp(logz, logx, logy))

####################################################################
####################################################################
from scipy.linalg import solve_continuous_lyapunov, svd
def balred(G, order = None, DCmatch = False, check = False, Tol = 1e-5):
    """
    Balanced truncation for state-space model reduction.
    https://stanford.edu/group/frg/course_work/CME345/CA-AA216-CME345-Ch6.pdf

    Parameters:
        G state space model (assumed SISO) - converted from tf form if given
        order (int): Desired order of the reduced model.
        check: (bool) check the results

    Returns:
        Gr: Reduced state-space system - returned in tf form if given in tf form  
    """    
    if not isinstance(G, control.StateSpace): # arrives as a TF
        convert_to_TF = True
    else:
        G = control.ss2tf(G)    
        convert_to_TF = False
       
    # find if there are any poles of G at origin
    G_trimmed = tf(G.num[0][0], np.trim_zeros(G.den[0][0], 'b'))
    number_cut = len(G.den[0][0]) - len(G_trimmed.den[0][0])
    #print(number_cut)
    
    # following done in SS form
    Gss = control.tf2ss(G_trimmed)    
    if order is None:
        order = Gss.A.shape[0] - 1
    order -= number_cut # account for poles cut out

    # Compute controllability Gramian
    Wc = solve_continuous_lyapunov(Gss.A, -Gss.B @ Gss.B.T) # solve AX + XA^H = Q 
    # Compute observability Gramian
    Wo = solve_continuous_lyapunov(Gss.A.T, -Gss.C.T @ Gss.C)

    U = np.linalg.cholesky(Wc)
    Z = np.linalg.cholesky(Wo)
    [W,S,Vh] = svd(U.T @ Z)
    S_sqrt_inv = np.linalg.inv(np.diag((np.sqrt(S))))

    T = S_sqrt_inv @ Vh @ Z.T 
    Ti = U @ W @ S_sqrt_inv

    Ab = T @ Gss.A @ Ti
    Bb = T @ Gss.B
    Cb = Gss.C @ Ti

    # Truncate to desired order
    Arr = Ab[:order, :order]
    Brr = Bb[:order, :]
    Crr = Cb[:, :order]
    Drr = Gss.D
    Gr = control.matlab.StateSpace(Arr, Brr, Crr, Drr)

    if DCmatch:
        if 1:  # DC gain matching - recommended but does not work well
            Are = Ab[:order, order:]
            Aer = Ab[order:, :order]
            Aee = Ab[order:, order:]
            Be = Bb[order:, :]
            Ce = Cb[:, order:]
            try:
                Aee_inv = np.linalg.inv(Aee)
                Arr -= Are @ Aee_inv @ Aer
                Brr -= Are @ Aee_inv @ Be
                Crr -= Ce @ Aee_inv @ Aer
                Drr -= Ce @ Aee_inv @ Be
            except: # singular matrix
                pass
            Gr = control.matlab.StateSpace(Arr, Brr, Crr, Drr)
        else: # DC gain matching - alternative
            Gr *= Gss.dcgain()/Gr.dcgain()

    def pretty_print(W,N=3):
        Wcprint = W
        Wcprint[np.abs(Wcprint) < Tol] = 0
        return np.round(Wcprint,N)

    if check:
        Wcb = solve_continuous_lyapunov(Gr.A, -Gr.B @ Gr.B.T) # solve AX + XA^H = Q 
        Wob = solve_continuous_lyapunov(Gr.A.T, -Gr.C.T @ Gr.C)
        print("\nControllability Gramian (Wc):\n", pretty_print(Wc))
        print("Observability Gramian (Wo):\n", pretty_print(Wo))
        print("\nBal Controllability Gramian (Wcb):\n", pretty_print(Wcb,6))
        print("Bal Observability Gramian (Wob):\n", pretty_print(Wob,6))

    # add the cut poles at zero back in - convert to TF form, add
    Gr = near_zero(control.ss2tf(Gr))*tf([1],[1, 0])**number_cut # TF

    return Gr if convert_to_TF else control.tf2ss(Gr) 

####################################################################

def pretty_row_print(X,msg = ''):
    print(msg + ', '.join('({0.real:.2f} + {0.imag:.2f}i)'.format(x) if np.iscomplex(x) else '{:.3f}'.format(x.real) for x in X))

####################################################################

def feedback_ff(G, K, Kff): 
    # polynomial level analysis to make sure that we a min order result
    """
    Feedback with feedforward control
    G: N/D
    K: Nc/Dc
    Kff: gain
    
    returns
    Gcl: (Kff+K)*G/(1+K*G)
    """
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

    # make sure the lengths are the same so we can add them
    max_len = max(len(DGDC), len(NGNC), len(NGDC))
    NGNC = np.pad(NGNC, (max_len - len(NGNC), 0), 'constant')
    NGDC = np.pad(NGDC, (max_len - len(NGDC), 0), 'constant')
    DGDC = np.pad(DGDC, (max_len - len(DGDC), 0), 'constant')

    return tf(Kff*NGDC+NGNC,DGDC+NGNC)

####################################################################
def writeGc(filename, Gc):
    '''
    Write the Gc to a file
    '''
    with open(filename, "w") as f:
        f.write(str(f"{np.real(-Gc.zeros())[0]:4.2f}"))
    with open(filename, "w") as f:
        f.write(str(f"{np.real(-Gc.poles())[0]:4.2f}"))
    with open(filename, "w") as f:
        gain = Gc.num[0][0][0]/Gc.den[0][0][0]
        f.write(str(f"{gain:4.2f}"))
       
####################################################################
if __name__ == "__main__":
    pass
else:
    print("This is a library of useful functions for the control systems in 16.06")