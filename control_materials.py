import numpy as np 
import cmath 
from sympy import Symbol, atan, pi, tan, sqrt, solve, log, symbols
from numpy.polynomial import Polynomial
from numpy import inf

from control.matlab import tf, rlocus, step, feedback, lsim, tf
from control import pade

import matplotlib
import matplotlib.pyplot as plt
r2d = 180/np.pi
tpi = 2*np.pi

# add break-in gains to improve plot
def Root_Locus_gains(L, Krange = np.logspace(-3, 3, num=1000), Tol = 1e-3, return_k_s = False):
    ''' 
    Augment the RL gains to include the break-in/break-out pts
    Thus leading to a better plot.

    In:     L(s)    - system for RL
            Krange  - set of K values

    Out:    Krange  - augmented set of K values

    '''
    #which RL
    RL_pos = True
    if max(Krange) <  0:
        RL_pos = False
    Krange = np.sort(np.append(Krange,0)) # add zero

    try:
        N = Polynomial(np.flip(L.num[0][0]))
        dNdx = N.deriv()
        D = Polynomial(np.flip(L.den[0][0]))
        dDdx = D.deriv()
        pd = dNdx*D - dDdx*N
        pdr = pd.roots() # candidate values of s for break-in/break-out pts

        Kkeep = [-1/np.real(L(x)) for x in pdr if abs(x.imag) < Tol] # k = -1/L(s) if s in pdr is real

        if 0:
            if len(L.num[0][0]) > 1: # confirm that dNds neq 0
                dNds = tf(np.flip(dNdx.coef),1)
                dDds = tf(np.flip(dDdx.coef),1)
                Kkeep = [-(dDds.horner(x.real)/dNds.horner(x.real))[0][0][0].real\
                         for x in pdr if abs(x.imag) < Tol]
            else:
                Kkeep = []
                n = len(L.den[0][0])
                syms = symbols('b:'+str(n-1))
                k = Symbol('k')
                for p in -pdr:
                    l = np.convolve([1, p],[*syms])
                    ll = [L.den[0][0][kk] - l[kk] for kk in range(len(l))]
                    ll[-1] += k
                    sol = solve(ll,[*syms,k],dict=True)
                    Kkeep = np.append(Kkeep,float(sol[0][k]))

        if RL_pos: # only look at the relevant sign K values depending on which RL is being drawn
            Kkeep = [x for x in Kkeep if ((x >= 0) and (x < inf))] 
        else:
            Kkeep = [x for x in Kkeep if ((x < 0) and (x > -inf))]
        if len(Kkeep) > 0:
            Krange = np.sort(np.append(Krange,Kkeep))
            for kk in Kkeep:
                Gcl_temp = feedback(L*kk,1)
                real_poles = [np.round(x.real,5) for x in Gcl_temp.poles() if abs(x.imag) < Tol]
                double_real_poles = set([x for x in real_poles if real_poles.count(x) > 1])
                print("\nFound breakin/out at K = {:4.3f}".format(kk))
                print("At possible locations s = "+', '.join('{:4.3f}'.format(x.real) for x in double_real_poles))
        else:
            double_real_poles = []
    except:
        print("Gain augmentation failed")
    return Krange #if return_k_s == False else Krange,Kkeep,double_real_poles
    
def RL_COM(L):
    ''' 
    Find the CoM of a RL for L(s)

    in:     L(s)    - system for RL

    out:    CoM     - Location of CoM    
            Ang     - angle of asymptotes in degrees

    if # poles = # zeros then avoids division by zero and returns None
    '''
    np = len(L.poles()) 
    nz = len(L.zeros())
    if (np >= nz+2):
        CoM = (sum([x for x in L.poles()]) - sum([x for x in L.zeros()]))/(np - nz)
        Ang = 180.0/(np - nz) % 360.0
    else:
        CoM = None
        Ang = None    
    return CoM, Ang

def Root_Locus_design_cancel(G, s_target = complex(-1,2), s_cancel = -1):
    '''
    RL Lead design of Gc by placing/canceling pole at s_cancel to ensure that CLP are at s_target
    '''
    phi_fromG = sum([cmath.phase(x) for x in (s_target - G.zeros())])*180/np.pi - \
                sum([cmath.phase(x) for x in (s_target - G.poles())])*180/np.pi

    Gczeros = np.array([s_cancel]) # cancel smallest plant real pole
    phi_from_Gc = sum([cmath.phase(x) for x in (s_target - Gczeros)])*180/np.pi
    phi_required = (180 + phi_fromG + phi_from_Gc)%360
    #print(phi_required)
    
    # now solve the phase condition equation for the comp pole location
    p = Symbol('p')
    Gcpoles = solve(atan(s_target.imag/(p + s_target.real)) - phi_required*pi/180,p)
    Gc = tf((1,-Gczeros[0]), (1,float(Gcpoles[0])))
    Gain = -1/np.real(G(s_target) * Gc(s_target))
    Gc *= Gain

    L = G*Gc    
    Gcl = feedback(L,1)

    return Gc, Gcl.poles()

def Root_Locus_design_ratio(G, s_target = complex(-1,2), gamma = 10):
    '''
    RL Lead design of Gc by to put CLP at s_target using a Gc.p/Gc.z = gamma
    '''
    z = Symbol('z')
    phi_fromG = sum([cmath.phase(x) for x in (s_target - G.zeros())])*180/np.pi - \
                sum([cmath.phase(x) for x in (s_target - G.poles())])*180/np.pi
    #print(phi_fromG)
    phi_required = (180 - phi_fromG)%360 
    #print(phi_required)

    # now solve the phase condition equation for the comp zero location
    # this implements the tan(A-B) = (tan(A) - tan(B)/(1+tan(A)tan(B)) condition in the notes 
    func = (s_target.imag/(z+s_target.real) - s_target.imag/(gamma*z+s_target.real))\
    /(1+(s_target.imag/(z+s_target.real))*(s_target.imag/(gamma*z+s_target.real)))
    Gczeros = max(solve(func - tan(phi_required*pi/180),z))
    Gcpoles = gamma*Gczeros

    Gc = tf((1, float(Gczeros)), (1,float(Gcpoles)))
    Gain = -1/np.real(G(s_target) * Gc(s_target))
    Gc *= Gain

    L = G*Gc
    Gcl = feedback(L,1)

    return Gc, Gcl.poles()
 
class Step_info:
    # init method or constructor
    def __init__(self,t,y, method = 0, t0 = 0, SettlingTimeLimits = [0.02], RiseTimeLimits = [0.1,0.9]):
        self.t = t
        self.y = y
        self.Yss = y[-1]
        self.SettlingTimeLimits = SettlingTimeLimits
        sgnYss = np.sign(self.Yss.real)

        tr_lower_index = np.where(sgnYss * (self.y - RiseTimeLimits[0] * self.Yss) >= 0)[0][0]
        tr_upper_index = np.where(sgnYss * (self.y - RiseTimeLimits[1] * self.Yss) >= 0)[0][0]
        self.Tr = self.t[tr_upper_index] - self.t[tr_lower_index]
        self.Tr_values = [self.t[tr_lower_index] - t0,self.t[tr_upper_index] - t0]

        settled = np.where(np.abs(self.y/self.Yss-1) >= SettlingTimeLimits)[0][-1]+1
        if settled < len(self.t):
            self.Ts = self.t[settled] - t0
        else:
            self.Ts = 0.  # avoids weird plot
        
        self.Mp = (self.y.max()/self.Yss-1)
        self.Tp = t[int(np.median(np.argwhere(self.y == self.y.max())))] - t0
        if method == 0: # which methods used to estimate zeta and wn from the step results
            print("Using Tp")
            self.zeta = 1/np.sqrt( 1 + (np.pi/np.log(self.Mp))**2 ) 
            self.wn = np.pi/self.Tp/np.sqrt(1-self.zeta**2)
        else:
            print("Using Ts")
            if self.SettlingTimeLimits[0] == 0.01:
                q = 4.6*self.Tp/np.pi/self.Ts # 1% rule
            else:
                q = 4*self.Tp/np.pi/self.Ts # 2 % rule
            self.zeta = q / np.sqrt( 1 + q**2 )
            self.wn = 4/self.Ts/self.zeta
    
    def printout(self):
        print("omega_n: \t%4.3f"%(self.wn))
        print("zeta   : \t%4.3f"%(self.zeta))
        print("Tr     : \t%4.2fs"%(self.Tr))
        print("Ts     : \t%4.2fs"%(self.Ts))
        print("Mp     : \t%4.2f"%(self.Mp))
        print("Tp     : \t%4.2fs"%(self.Tp))
        print("Yss    : \t%4.2f"%(self.Yss))

        #for k in S:
        #    if k in ['RiseTime','Overshoot','PeakTime','SettlingTime']:
        #        try:
        #            print(f"{k}: {S[k]:3.4}")
        #        except:
        #            print(f"{k}: {S[k]:4}")

        
    def nice_plot(self,ax):
        print(f"Using {self.SettlingTimeLimits[0] = :4.2f}")
        ylim=(np.floor(np.min(self.y)),np.ceil(10.*np.max(self.y))/10.0)
        ax.plot(self.t,self.y,'b')
        ymax = np.max(ylim) # needed for plot scaling

        ax.axvline(x = self.Tr_values[0],ymax=0.1*self.Yss/ymax,c='r',ls='dashed')
        ax.axvline(x = self.Tr_values[1],ymax=0.9*self.Yss/ymax,c='r',ls='dashed')
        ax.axvline(x = self.Ts,c='grey',ls='dashed')
        ax.axhline(y = (1+self.SettlingTimeLimits[0])*self.Yss,c='grey',ls='dashed',lw=1)
        ax.axhline(y = (1-self.SettlingTimeLimits[0])*self.Yss,c='grey',ls='dashed',lw=1)
        ax.axhline(y = self.Yss*(1 + self.Mp), xmin=0, xmax=self.Tp/max(self.t), c='green',ls='dashed',lw=2)
        ax.axvline(ymax = self.Yss*(1 + self.Mp)/ymax, x=self.Tp, c='m',ls='dashed',lw=2)
        ax.text(self.Tr/2,0.25*self.Yss,"Tr = {0:.2f}".format(self.Tr))
        ax.text(self.Tp,0.75*self.Yss,"Tp = {0:.2f}".format(self.Tp))
        ax.text(self.Ts,0.5*self.Yss,"Ts = {0:.2f}".format(self.Ts))
        ax.text(self.Tp*1.1,self.Yss*(1 + self.Mp),"Mp = {0:.2f}".format(self.Mp))
        ax.set_xlabel('time [s]')
        ax.set_ylabel('Response')
        ax.set_title('Step Response')
        ax.set_ylim(ylim)
        ax.set_xlim([0,max(self.t)])

def lead_design(G, wc_des = 1, PM = 45):
    j = complex(0,1)
    Gf = G(j*wc_des)
    phi_G = cmath.phase(Gf)*r2d
    phi_m = (PM - (180 + phi_G))/r2d # robust?
    zdp = (1-np.sin(phi_m))/(1+np.sin(phi_m))
    z = np.sqrt(wc_des**2*zdp)
    p = z/zdp
    Gc_lead = tf([1, z],[1, p]) 
    L = G*Gc_lead
    k_c = 1/np.abs(L(j*wc_des))
    Gc_lead *= k_c
    return Gc_lead

def lag_design(gain_inc = 10, gamma = 10, wc = 1):
    # gain_inc: goal for adding the lag
    # gamma = 10: heuristic design approach
    # wc: design goal
    zl = wc/gamma 
    pl = zl/gain_inc
    Gc_lag = tf([1, zl],[1, pl]) # lag comp
    return Gc_lag 
                
def find_wc(omega, G, mag = 1):
    return np.interp(mag,np.flipud(np.abs(G)),np.flipud(omega))

def pshift(Gp):
    while (np.max(Gp) < -np.pi):
        Gp += 2*np.pi
    while (np.min(Gp) > np.pi):
        Gp -= 2*np.pi
    return Gp
    
SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

def caption(txt,fig, xloc=0.5, yloc=-0.05):
    fig.text(xloc, yloc, txt, ha='center',size=MEDIUM_SIZE,color='blue')

def my_pzmap(G,ax):
    ax.plot(np.real(G.poles()),np.imag(G.poles()),'bx',ms=8,markerfacecolor=None)
    ax.plot(np.real(G.zeros()),np.imag(G.zeros()),'o',ms=8,markeredgewidth=2, markeredgecolor='r',markerfacecolor=(1, 0, 0.2, 0.1))
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
