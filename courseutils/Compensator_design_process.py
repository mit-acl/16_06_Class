from pathlib import Path
import sys

# repo_root/16_06_Class/notebooks → repo_root/16_06_Class
repo_root = Path.cwd().parents[0]
sys.path.insert(0, str(repo_root / "16_06_Class"))

import courseutils.basic_material as bm
import courseutils.control_materials as cm

j = complex(0,1)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import control as ct
import control.matlab as cmat
from IPython.display import display, Math, Markdown
from types import SimpleNamespace

def time_to_pole_specs(Mp, Tr, Tp, Ts, verbose=False):
    ''' Convert time-domain specifications to pole specifications. '''
    info = []

    if Mp is not None:
        zeta = np.sqrt(np.log(Mp)**2 / (np.pi**2 + np.log(Mp)**2))
        info.append(
        rf"The damping ratio $\zeta$ is obtained from the percent overshoot specification $M_p$ using "
        rf"\[M_p = e^{{-\frac{{\pi \zeta}}{{\sqrt{{1-\zeta^2}}}}}}\quad \Rightarrow \quad\zeta  = \frac{{|\ln M_p|}}{{\sqrt{{\pi^2 + (\ln M_p)^2}}}} = \frac{{|\ln {Mp:.3f}|}}{{\sqrt{{\pi^2 + (\ln {Mp:.3f})^2}}}} = {zeta:.3f}.\]")
    else:
        zeta = None

    if Tr is not None:
        wn = 1.8 / Tr
        info.append(
            rf"The natural frequency $\omega_n$ is estimated from the rise time $T_r$ using the approximation "
            rf"\[T_r \approx \frac{{1.8}}{{\omega_n}} \Rightarrow \omega_n \approx \frac{{1.8}}{{T_r}} = \frac{{1.8}}{{{Tr:.3f}}} = {wn:.3f}.\]")
    else:
        wn = None

    if Tp is not None:
        wd = np.pi / Tp
        info.append(
            r"The damped natural frequency $\omega_d$ is obtained from the peak time $T_p$ using "
            r"\[T_p = \frac{\pi}{\omega_d}\quad \Rightarrow \quad\omega_d = \frac{\pi}{T_p} .\]")
    else:
        wd = None

    if Ts is not None:
        sigma = 4 / Ts
        info.append(
            r"The real part of the dominant pole $\sigma$ is estimated from the settling time $T_s$ using "
            r"\[T_s \approx \frac{4}{\sigma}\quad \Rightarrow \quad\sigma \approx \frac{4}{T_s} .\]")
    else:
        sigma = None

    # --- resolve overlapping constraints for wn ---
    if wn is None:
        if (wd is not None) and (zeta is not None):
            wn = wd / np.sqrt(1 - zeta**2)
            info.append(
                r"When $\omega_d$ and $\zeta$ are known, the natural frequency is recovered using "
                r"\[\omega_d = \omega_n \sqrt{1-\zeta^2}\quad \Rightarrow \quad\omega_n = \frac{\omega_d}{\sqrt{1-\zeta^2}} .\]")
        elif (sigma is not None) and (zeta is not None):
            wn = sigma / zeta
            info.append(
                r"When $\sigma$ and $\zeta$ are known, the natural frequency follows from "
                r"\[\sigma = \zeta \omega_n\quad \Rightarrow \quad\omega_n = \frac{\sigma}{\zeta} .\]")
        else:
            return None, None, None
    else:
        if (wd is not None) and (zeta is not None):
            wn_test = wd / np.sqrt(1 - zeta**2)
            wn = max(wn, wn_test)
        elif (sigma is not None) and (zeta is not None):
            wn_test = sigma / zeta
            wn = max(wn, wn_test)

    # --- resolve overlapping constraints for zeta ---
    if zeta is None:
        if (wd is not None) and (wn is not None):
            zeta = np.sqrt(1 - (wd / wn)**2)
            info.append(
                r"When $\omega_d$ and $\omega_n$ are known, the damping ratio is obtained from "
                r"\[\omega_d = \omega_n \sqrt{1-\zeta^2}\quad \Rightarrow \quad\zeta = \sqrt{1-\left(\frac{\omega_d}{\omega_n}\right)^2} .\]")
        elif (sigma is not None) and (wn is not None):
            zeta = sigma / wn
            info.append(
                r"When $\sigma$ and $\omega_n$ are known, the damping ratio follows from "
                r"\[\sigma = \zeta \omega_n\quad \Rightarrow \quad\zeta = \frac{\sigma}{\omega_n} .\]")
        else:
            return None, None, None
    else:
        if (wd is not None) and (wn is not None):
            zeta_test = np.sqrt(1 - (wd / wn)**2)
            zeta = max(zeta, zeta_test)
        elif (sigma is not None) and (wn is not None):
            zeta_test = sigma / wn
            zeta = max(zeta, zeta_test)

    # resulting damped freq
    wdf = wn*np.sqrt(1-zeta**2)
    sigma = zeta*wn

    items = [
        (r"$M_p$", Mp),
        (r"$T_r$", Tr),
        (r"$T_p$", Tp),
        (r"$T_s$", Ts),
        (r"$\omega_n$", wn),
        (r"$\zeta$", zeta),
        (r"$\omega_d$", wdf),
        (r"$\sigma$", sigma),
    ]

    parts = []
    for k, v in items:
        if v is not None:
            if isinstance(v, (int, float, np.floating)):
                parts.append(f"{k}: {v:4.2f}")
            else:
                parts.append(f"{k}: {v}")

    msgs = r"In summary, " + ", ".join(parts)
    display(Markdown(msgs))
    info.append(msgs)

    msgs = rf"As a result, the target pole locations are $s_{{target}}={-sigma:.3f} \pm {wdf:.3f}i$."
    info.append(msgs)
    
    info_str = None
    if verbose:
        info_str = "\n\n".join(info)

    return wn, zeta, info_str

def design_process(G, Mp = None, Tr = None, Tp = None, Ts = None, ess_step = None, 
    ess_ramp = np.inf, gamma = 10, max_iter = 1, make_plots = False, 
    file_prefix = "Rec6_des", s_cancel=None, verbose=False):
    ''' design_process(G, Mp = None, Tr = None, Tp = None, Ts = None, ess_step = None, ess_ramp = np.inf, 
        gamma = 10, max_iter = 1, make_plots = False)
    
    Design a compensator for the system G based on given specifications.
    Parameters:
        G: Transfer function of the plant   
        Mp: Maximum overshoot (default: None)
        Tr: Rise time (default: None)
        Tp: Peak time (default: None)
        Ts: Settling time (default: None)
        ess_step: Steady-state error for step input (default: None)
        ess_ramp: Steady-state error for ramp input (default: np.inf)
        gamma: Gain for design (default: 10)
        max_iter: Maximum number of iterations for design (default: 1)
        make_plots: Boolean to enable plotting (default: False)
    Returns:
        Gc: Compensator transfer function
        Gc_lead: Lead compensator transfer function
        Gc_lag: Lag compensator transfer function
    '''
    info = SimpleNamespace()
    info.error = None
    info.latex = None
    info.lag_latex = None

    Gc_lead = None
    Gc_lag = ct.tf(1,1)
    G_type = cm.system_type(G)

    check_ramp = False
    if ess_ramp < np.inf:
        print("Ramp steady state error specified")
        check_ramp = True        

    wn, zeta, info_str = time_to_pole_specs(Mp, Tr, Tp, Ts, verbose=verbose)
    info.latex = info_str
    if wn is None or zeta is None:
        msgs = "Error in converting time specs to pole specs"
        print(msgs)
        info.error = msgs
        return info

    s_target = complex(-zeta*wn,wn*np.sqrt(1-zeta**2))
    scale = np.abs(s_target.real)
    cm.pretty_row_print([s_target],"Target Pole: ")

    if s_cancel is None:
        s_cancel = s_target.real
    
    #################################################################################################
    #################################################################################################
    if (G_type == 0 and check_ramp): # must add integrator
        print("\nAdding integrator to plant for design")
        G_pert = ct.tf(1,[1,0])
    else:
        G_pert = ct.tf(1,1)

    for iter_count in range(max_iter):
        if Gc_lag is None:
            Gc_lag = ct.tf(1,1)
        print("\nDesiging lead compensator")
        Gc_lead, _, lead_info = cm.Root_Locus_design_cancel(G*G_pert*Gc_lag, s_target = s_target, s_cancel = s_cancel, verbose=True)
        Gc_lead *= G_pert # if integrator needed, then add it to the lead compensator
        cm.show_tf_latex(Gc_lead, f"G_{{c_{{lead}}}}^{{(iter {iter_count+1})}}",show=True,factor=True)
        L_lead = G * Gc_lead
        
        Gcl_lead = ct.feedback(L_lead,1)
        scl_lead = Gcl_lead.poles()
        print(f"\nLead Zero: {Gc_lead.zeros()[0]:.2f}")
        print(f"Lead Pole: {Gc_lead.poles()[0]:.2f}") 
        cm.pretty_row_print(scl_lead,"Lead CLP ")
        
        if cm.system_type(L_lead) < 1:
            ess_step_orig = find_ess(L_lead, type='step')
        else:
            ess_step_orig = 0
        print(f"\nStep steady state error for lead design -- e_ss step: {ess_step_orig:.3f}")    

        if check_ramp:
            ess_ramp_orig = find_ess(L_lead, type = 'ramp')
            print(f"e_ss ramp: {ess_ramp_orig:.2f}")    

        # assumes we will only add 1 lag for either ramp or step error correction
        if (ess_step is not None) and (ess_step_orig > ess_step):
            print("\nAdding lag compensator for step error")
            Kp_desired = 1/ess_step - 1
            Kp_lead = cm.find_Kp(L_lead)
            Kp_ratio = Kp_desired/Kp_lead

            z_lag = np.abs(s_target/gamma)
            p_lag = z_lag/Kp_ratio
            Gc_lag = ct.tf([1,z_lag],[1,p_lag])
            
            Gc = Gc_lead * Gc_lag
            L_lag = L_lead * Gc_lag        
            Gcl_lag = ct.feedback(L_lag,1)
            scl_lag = Gcl_lag.poles()

            print(f"\nKp_lead: {Kp_lead:.2f}")
            print(f"Kp desired: {Kp_desired:.2f}")
            print(f"Kp ratio: {Kp_ratio:.2f}")
            print(f"\nLag Zero: {z_lag:.2f}")
            print(f"Lag Pole: {p_lag:.2f}") 
            print(f"Kp_lag: {cm.find_Kp(L_lag):.2f}")
            cm.pretty_row_print(scl_lag,"Lag CLP ")

            lag_latex_paragraph = (
                rf"The desired step steady-state error of $e_{{ss}} = {ess_step:.2f}$ requires "
                rf"$K_p^{{\mathrm{{desired}}}} = {Kp_desired:.2f}$, whereas the current system has "
                rf"$K_p = {Kp_lead:.2f}$, requiring an increase by a ratio of "
                rf"$\beta = {Kp_ratio:.2f}$. "
                rf"Using a lag compensator with zero -- pole separation factor $\gamma = {gamma:.1f}$, "
                rf"the zero is placed at "
                rf"$z_l = \frac{{|s_{{target}}|}}{{\gamma}} = {z_lag:.2f}$ "
                rf"and the pole at "
                rf"$p_l = \frac{{z_l}}{{\beta}} = {p_lag:.2f}$. "
                rf"The resulting lag compensator transfer function is "
                rf"$$G_c^{{\mathrm{{lag}}}}(s) = \dfrac{{s + {-z_lag:.3f}}}{{s + {-p_lag:.3f}}}.$$"
            )
            info.lag_latex = lag_latex_paragraph

            if check_ramp:
                ess_ramp_lag = find_ess(L_lag, type = 'ramp')
                print(f"e_ss ramp: {ess_ramp_lag:.3f}")    
            else:
                ess_step_lag = find_ess(L_lag, type='step')
                print(f"e_ss step: {ess_step_lag:.3f}")    
                
        elif check_ramp and ess_ramp_orig > ess_ramp:
            print("\nAdding lag compensator for Ramp error")
            Kv_desired = 1/ess_ramp 
            Kv_lead = cm.find_Kv(L_lead)
            Kv_ratio = Kv_desired/Kv_lead

            z_lag = np.abs(s_target/gamma)
            p_lag = z_lag/Kv_ratio
            Gc_lag = ct.tf([1,z_lag],[1,p_lag])
            
            Gc = Gc_lead * Gc_lag
            L_lag = L_lead * Gc_lag        
            Gcl_lag = ct.feedback(L_lag,1)
            scl_lag = Gcl_lag.poles()

            print(f"Kv_lead: {Kv_lead:.2f}")
            print(f"Kv desired: {Kv_desired:.2f}")
            print(f"Kv ratio: {Kv_ratio:.2f}\n")
            print(f"Lag Zero: {z_lag:.2f}")
            print(f"Lag Pole: {p_lag:.2f}") 
            print(f"Kv_lag: {cm.find_Kv(L_lag):.2f}")
            cm.pretty_row_print(scl_lag,"Lag CLP")

            ess_step_lag = 0
            print(f"e_ss step: {ess_step_lag:.3f}")    
            ess_ramp_lag = find_ess(L_lag, type = 'ramp')
            print(f"e_ss ramp: {ess_ramp_lag:.3f}")    
        else:
            print("No lag compensator needed")
            Gc_lag = None
            Gc = Gc_lead
            L_lag = L_lead        
            Gcl_lag = Gcl_lead
            scl_lag = scl_lead          

        # Now Plot the design and response
        # RL
        fig, ax1 = plt.subplots(1,figsize=(5, 5),dpi=150,constrained_layout = True)
        rl =  ct.rlocus(L_lag, gains=cm.Root_Locus_gains(L_lag,np.logspace(-3, 1, num=1500)),color='b',ax=ax1)
        ax1.plot(scl_lag.real,scl_lag.imag,'cs',ms=8)
        ax1.plot(scl_lag.real,-scl_lag.imag,'cs',ms=8)
        ax1.plot(s_target.real,s_target.imag,'md',ms=8)
        ax1.plot(s_target.real,-s_target.imag,'md',ms=8)
        ax1.set_xlabel('Real')
        ax1.set_ylabel('Imaginary')
        ax1.set_title('K > 0',loc='left')
        try:
            plt.title(r'$\gamma={:3.1f}$'.format(gamma),loc='right')
        except:
            pass
        ax1.plot(np.real(Gc_lead.zeros()), np.imag(Gc_lead.zeros()), 'go', ms=8, label='Lead Zero')
        ax1.plot(np.real(Gc_lead.poles()), np.imag(Gc_lead.poles()), 'rx', ms=8, label='Lead Pole')
        if Gc_lag is not None:
            ax1.plot(np.real(Gc_lag.zeros()), np.imag(Gc_lag.zeros()), 'yo', ms=8, label='Lag Zero')
            ax1.plot(np.real(Gc_lag.poles()), np.imag(Gc_lag.poles()), 'mx', ms=8, label='Lag Pole')

        ax1.text(0.05, 0.95, f'Lead Zero: {np.real(-Gc_lead.zeros())[0]:.2f}', transform=ax1.transAxes, fontsize=12, color='green', verticalalignment='top')
        ax1.text(0.05, 0.90, f'Lead Pole: {np.real(-Gc_lead.poles())[0]:.2f}', transform=ax1.transAxes, fontsize=12, color='red', verticalalignment='top')
        ax1.text(0.05, 0.85, f'Lead Gain: {Gc_lead.num[0][0][0]/Gc_lead.den[0][0][0]:.2f}', transform=ax1.transAxes, fontsize=12, color='blue', verticalalignment='top')
        if Gc_lag is not None:
            ax1.text(0.05, 0.75, f'Lag Zero: {np.real(-Gc_lag.zeros())[0]:.2f}', transform=ax1.transAxes, fontsize=12, color='black', verticalalignment='top')
            ax1.text(0.05, 0.7, f'Lag Pole: {np.real(-Gc_lag.poles())[0]:.2f}', transform=ax1.transAxes, fontsize=12, color='magenta', verticalalignment='top')

        # Insert to lower right of ax1 that plots the root locus near the origin
        ax_inset = ax1.inset_axes([0.75, 0.15, 0.2, 0.2])
        rl =  ct.rlocus(L_lag, gains=cm.Root_Locus_gains(L_lag, np.logspace(-3, 1, num=1500)), color='b', ax=ax_inset)
        ax_inset.plot(s_target.real, s_target.imag, 'md', ms=4)
        ax_inset.plot(s_target.real, -s_target.imag, 'md', ms=4)
        ax_inset.plot(scl_lag.real, scl_lag.imag, 'cs', ms=4)
        ax_inset.plot(scl_lag.real, -scl_lag.imag, 'cs', ms=4)
        ax_inset.plot(np.real(Gc_lead.zeros()), np.imag(Gc_lead.zeros()), 'go', ms=4, label='Lead Zero')
        ax_inset.plot(np.real(Gc_lead.poles()), np.imag(Gc_lead.poles()), 'rx', ms=4, label='Lead Pole')

        if Gc_lag is not None:
            ax_inset.plot(np.real(Gc_lag.zeros()), np.imag(Gc_lag.zeros()), 'yo', ms=4, label='Lag Zero')
            ax_inset.plot(np.real(Gc_lag.poles()), np.imag(Gc_lag.poles()), 'mx', ms=4, label='Lag Pole')

        scale = np.ceil(np.abs(s_target.real))*3
        offset = np.sum(G.poles().real)/len(G.poles())
        ax_inset.axis('equal')
        ax_inset.set_xlim((-scale/5+s_target.real,scale/5+s_target.real))
        ax_inset.set_ylim([-scale/5, scale/5])
        bm.nicegrid(ax_inset)

        ax1.axis('equal')
        ax1.set_xlim((-scale+s_target.real,scale+s_target.real))
        ax1.set_ylim([-scale, scale])
        # Create custom legend handles

        if Gc_lag is None:
            custom_lines = [
                Line2D([0], [0], color='blue', lw=2, linestyle='-'),
                Line2D([0], [0], color='magenta', marker='d', markersize=8, linestyle='None'),
                Line2D([0], [0], color='cyan', marker='s', markersize=8, linestyle='None'),
                Line2D([0], [0], color='green', marker='o', markersize=8, linestyle='None'),
                Line2D([0], [0], color='red', marker='x', markersize=8, linestyle='None'),
            ]
            # Add legend with custom handles
            ax1.legend(custom_lines, ['Root Locus', 'Desired CLP', 'Lag CLP', 'Lead Zero', 'Lead Pole'], loc='upper right')
        else:
            custom_lines = [
                Line2D([0], [0], color='blue', lw=2, linestyle='-'),
                Line2D([0], [0], color='magenta', marker='d', markersize=8, linestyle='None'),
                Line2D([0], [0], color='cyan', marker='s', markersize=8, linestyle='None'),
                Line2D([0], [0], color='green', marker='o', markersize=8, linestyle='None'),
                Line2D([0], [0], color='red', marker='x', markersize=8, linestyle='None'),
                Line2D([0], [0], color='yellow', marker='o', markersize=8, linestyle='None'),
                Line2D([0], [0], color='magenta', marker='x', markersize=8, linestyle='None'),
            ]
            # Add legend with custom handles
            ax1.legend(custom_lines, ['Root Locus', 'Desired CLP', 'Lag CLP', 'Lead Zero', 'Lead Pole', 'Lag Zero', 'Lag Pole'], loc='upper right')

        bm.nicegrid(ax1)
        if make_plots:
            plt.savefig("./figs/"+file_prefix+str(iter_count)+"_RL.pdf", dpi=600)

        # Plot step response
        Ts_pred = np.abs(4/(np.real(s_target)))
        t = cmat.linspace(0, np.ceil(10*Ts_pred), 500) # long to y_ss right
        y_lag,t_lag = cmat.step(ct.tf2ss(Gcl_lag),T=t)

        fig, ax2 = plt.subplots(1,figsize=(8,5),dpi=150,constrained_layout = True)
        ax2.plot(t_lag, y_lag,'b')
        ST = cm.Step_info(t_lag, y_lag)
        ST.nice_plot(ax2,Tmax=np.ceil(2*Ts_pred))
        ax2.set_xlim(0, np.ceil(2*Ts_pred))
        ax2.set_title('Step Response of Gcl')
        ax2.set_title(f'Design {iter_count}', loc='left')

        if make_plots:
            plt.savefig("./figs/"+file_prefix+str(iter_count)+"_step.pdf", dpi=600)

        if check_ramp:
            factor = 100
            fig, ax3 = plt.subplots(1,figsize=(8,5),dpi=150)
            Tramp_pred = np.abs(4/(np.real(s_target)))
            # Plot step response
            t = cmat.linspace(0, int(10*Tramp_pred), 1000)
            y_lead_ramp,t_lead_ramp, _ = cmat.lsim(ct.tf2ss(Gcl_lead),U=t,T=t)
            y_lag_ramp,t_lag_ramp,_ = cmat.lsim(ct.tf2ss(Gcl_lag),U=t,T=t)
            ax3.plot(t_lag_ramp, y_lag_ramp,'b')
            ax3.plot(t_lag_ramp, factor*(t_lag_ramp-y_lag_ramp),'b--')
            ax3.plot(t_lead_ramp, y_lead_ramp,'r-')
            ax3.plot(t_lead_ramp, factor*(t_lead_ramp-y_lead_ramp),'r--')
            ax3.set_title('Ramp Response of Gcl')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Response')
            ax3.grid(True)
            ax3.axhline(y=factor*ess_ramp, color='k', linestyle='--')
            ax3.legend(('Lag','Lag Err','Lead','Lead Err'))
            ax3.set_title(f'Design {iter_count}', loc='left')
            if make_plots:
                plt.savefig("./figs/"+file_prefix+str(iter_count)+"_ramp_lag.pdf", dpi=600)

        plt.show()

        cm.show_tf_latex(Gc, "G_c(s)",show=True,factor=True)
        cm.show_tf_latex(Gc_lead, "G_{c_{lead}}(s)",show=True,factor=True)
        if Gc_lag is not None:
            cm.show_tf_latex(Gc_lag, "G_{c_{lag}}(s)",show=True,factor=True)

        info.Gc_lead = Gc_lead
        info.lead_info = lead_info
        info.Gc_lag = Gc_lag
        info.Gc = Gc
    return info

def find_ess(G, type = 'step'):
    if type == 'step':
        K = cm.find_Kp(G)
        if K is not None:
            return 1/(1+K)
        else:
            print('Kp error')
    elif type == 'ramp':
        K = cm.find_Kv(G)
        if np.abs(K) > 0:
            return 1/K
        else:
            print('Kv error')
    print("Unknown type")
    return None

if __name__ == "__main__":
    # Example usage:
    print("G = ct.tf([10],[1, 15, 50, 0])")
    print("info = design_process(G, Mp=0.2, Ts=1, ess_ramp=0.05, gamma=10, max_iter=2, make_plots=True)")  
