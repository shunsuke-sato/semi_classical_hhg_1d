import numpy as np
from matplotlib import pyplot as plt
from numba import njit, prange

## Parameters
ev = 27.2114
fs = 0.024189
a_b = 0.529177
eps_g = 4.18/ev
mu_eh = 0.083

boundary_1st_k = np.pi/8.0
boundary_2nd_k = 2*np.pi/8.0

dt_traj = 1.0
tprop = 120.0/fs

## Laser field parameters
omega = 0.387/ev
#F0 = 0.165*a_b/ev

tau = 9*2*np.pi/omega

dt_ex = tau/200

@njit(cache=True)
def A_field_t(t, F0):
    if t < 0.0 or t >= tau:
        return 0.0
    s = np.sin(np.pi * t / tau)
    return (F0 / omega) * np.sin(omega * t) * s * s * s * s

#def A_field_t(t, F0):
#    """
#    Returns the vector potential of the laser field at time t.
#    """
#
#    A0 = F0/omega
#
#    if t < 0:
#        return 0.0
#    elif t < tau:
#        return A0 * np.sin(omega * t) * np.sin(np.pi * t / tau)**4
#    else:
#        return 0.0

    
def eps_kane_band(k):
    """
    Returns the Kane band energy at wavevector k.
    """
    return eps_g*np.sqrt(1.0 + k**2/(mu_eh*eps_g))

def velocity_kane_band(k):
    """
    Returns the velocity of the Kane band at wavevector k.
    """
    return (k/mu_eh) / np.sqrt(1.0 + k**2/(mu_eh*eps_g))


def analyze_non_scattered_trajectory(tex, F0):
    """
    Analyzes the non-scattered trajectory of an electron in a laser field.
    """

    k0 = -A_field_t(tex, F0)
    k = 0.0
    x = 0.0
    t = tex


    max_energy = -1.0

    while t < tprop:
        k = k0 + A_field_t(t+dt_traj*0.5, F0)
        v = velocity_kane_band(k)
        x_new = x + v*dt_traj
        if(x_new == 0 or x_new*x <0):
            energy = eps_kane_band(k)
            if energy > max_energy:
                max_energy = energy

        x = x_new
        t += dt_traj

    
    return max_energy

def analyze_singly_scattered_trajectory(tex, F0):
    """
    Analyzes the singly scattered trajectory of an electron in a laser field.
    """

    nskip = 4
    k0 = -A_field_t(tex, F0)

    max_energy = -1.0


    for iskip in range(nskip):
        icount = 0
        k = 0.0
        x = 0.0
        t = tex

        kmom_scattering = 0.0
        if_any_scattering = False
        
        while t < tprop:
            k_new = k0 + A_field_t(t+dt_traj*0.5, F0) + kmom_scattering

            if (if_any_scattering == False):
                icount, kmom_scattering = check_scattering_event(
                    k, k_new, iskip, icount, kmom_scattering)
                
                if icount > iskip:
                    if_any_scattering = True

            k = k0 + A_field_t(t+dt_traj*0.5, F0) + kmom_scattering

            v = velocity_kane_band(k)
            x_new = x + v*dt_traj

            if(if_any_scattering and (x_new == 0 or x_new*x <0)):
                energy = eps_kane_band(k)
                if energy > max_energy:
                    max_energy = energy

            x = x_new
            t += dt_traj


    return max_energy

def analyze_doubly_scattered_trajectory(tex, F0):
    """
    Analyzes the doubly scattered trajectory of an electron in a laser field.
    """

    nskip_1st = 4
    nskip_2nd = 4

    k0 = -A_field_t(tex, F0)

    max_energy = -1.0


    for iskip in range(nskip_1st):
        for jskip in range(nskip_2nd):

            if_any_scattering_1st = False
            if_any_scattering_2nd = False

            icount_1st = 0
            icount_2nd = 0

            k = 0.0
            x = 0.0
            t = tex

            kmom_scattering_1st = 0.0
            kmom_scattering_2nd = 0.0
        
            while t < tprop:
                k_new = k0 + A_field_t(t+dt_traj*0.5, F0) \
                    + kmom_scattering_1st + kmom_scattering_2nd

                if(if_any_scattering_1st == False):

                    icount_1st, kmom_scattering_1st = check_scattering_event(
                        k, k_new, iskip, icount_1st, kmom_scattering_1st)

                    if icount_1st > iskip:
                        if_any_scattering_1st = True

                else:

                    if(if_any_scattering_2nd == False):
                        icount_2nd, kmom_scattering_2nd = check_scattering_event(
                            k, k_new, jskip, icount_2nd, kmom_scattering_2nd)

                        if icount_2nd > jskip:
                            if_any_scattering_2nd = True



                k = k0 + A_field_t(t+dt_traj*0.5, F0) \
                    + kmom_scattering_1st + kmom_scattering_2nd

                v = velocity_kane_band(k)
                x_new = x + v*dt_traj
                if(if_any_scattering_1st and if_any_scattering_2nd 
                   and (x_new == 0 or x_new*x <0)):
                    energy = eps_kane_band(k)
                    if energy > max_energy:
                        max_energy = energy

                x = x_new
                t += dt_traj


    return max_energy

def check_scattering_event(k_old, k_new, iskip, icount, kmom_scattering):
    
    if_scattering = 0

    
    if (k_new == boundary_1st_k or (k_old-boundary_1st_k)*(k_new-boundary_1st_k) < 0):
        if_scattering = 1
        kmom_scattering_tmp = -2*boundary_1st_k
    elif (k_new == boundary_2nd_k or (k_old-boundary_2nd_k)*(k_new-boundary_2nd_k) < 0):
        if_scattering = 1
        kmom_scattering_tmp = -2*boundary_2nd_k
    elif (k_new == -boundary_1st_k or (k_old+boundary_1st_k)*(k_new+boundary_1st_k) < 0):
        if_scattering = 1
        kmom_scattering_tmp = 2*boundary_1st_k
    elif (k_new == -boundary_2nd_k or (k_old+boundary_2nd_k)*(k_new+boundary_2nd_k) < 0):
        if_scattering = 1
        kmom_scattering_tmp = 2*boundary_2nd_k


    if if_scattering == 1:
        if  icount == iskip:
            kmom_scattering += kmom_scattering_tmp
        
        icount += 1

    return icount, kmom_scattering
            
def run_non_scattered_trajectory_analysis(F0):
    """
    Runs the analysis of a single scattered trajectory.
    """

    time = np.arange(0, tau, dt_ex)
    max_emission_energy_list = []
    time_list = []

    for tex in time:
        max_emission_energy = analyze_non_scattered_trajectory(tex, F0)
        if max_emission_energy > 0:
            max_emission_energy_list.append(max_emission_energy)
            time_list.append(tex)



    if len(max_emission_energy_list) == 0:
        max_emission_energy_among_all_time = -1.0
    else:
        max_emission_energy_among_all_time = np.max(max_emission_energy_list)


    return max_emission_energy_among_all_time

def run_singly_scattered_trajectory_analysis(F0):
    """
    Runs the analysis of a single scattered trajectory.
    """

    time = np.arange(0, tau, dt_ex)
    max_emission_energy_list = []
    time_list = []

    for tex in time:
        max_emission_energy = analyze_singly_scattered_trajectory(tex, F0)
        if max_emission_energy > 0:
            max_emission_energy_list.append(max_emission_energy)
            time_list.append(tex)



    if len(max_emission_energy_list) == 0:
        max_emission_energy_among_all_time = -1.0
    else:
        max_emission_energy_among_all_time = np.max(max_emission_energy_list)


    return max_emission_energy_among_all_time

def run_doubly_scattered_trajectory_analysis(F0):
    """
    Runs the analysis of a doubly scattered trajectory.
    """

    time = np.arange(0, tau, dt_ex)
    max_emission_energy_list = []
    time_list = []

    for tex in time:
        max_emission_energy = analyze_doubly_scattered_trajectory(tex, F0)
        if max_emission_energy > 0:
            max_emission_energy_list.append(max_emission_energy)
            time_list.append(tex)



    if len(max_emission_energy_list) == 0:
        max_emission_energy_among_all_time = -1.0
    else:
        max_emission_energy_among_all_time = np.max(max_emission_energy_list)


    return max_emission_energy_among_all_time

def field_strength_scan():
    """
    Scans the field strength and analyzes the scattered trajectories.
    """

    F0_VpAA_list = np.linspace(0.01, 0.25, 60)
    F0_list = F0_VpAA_list*a_b/ev


    F0_append_list = []
    max_emission_energy_vs_F0 = []

    F0_append_list_sngle_scattering = []
    max_emission_energy_vs_F0_single_scattering = []

    F0_append_list_double_scattering = []
    max_emission_energy_vs_F0_double_scattering = []

    for F0 in F0_list:
        max_emission_energy_among_all_time = run_non_scattered_trajectory_analysis(F0)
        if max_emission_energy_among_all_time > 0:
            F0_append_list.append(F0)
            max_emission_energy_vs_F0.append(max_emission_energy_among_all_time)

        max_emission_energy_among_all_time_single_scattering = run_singly_scattered_trajectory_analysis(F0)
        if max_emission_energy_among_all_time_single_scattering > 0:
            F0_append_list_sngle_scattering.append(F0)
            max_emission_energy_vs_F0_single_scattering.append(max_emission_energy_among_all_time_single_scattering)


        max_emission_energy_among_all_time_double_scattering = run_doubly_scattered_trajectory_analysis(F0)
        if max_emission_energy_among_all_time_double_scattering > 0:
            F0_append_list_double_scattering.append(F0)
            max_emission_energy_vs_F0_double_scattering.append(max_emission_energy_among_all_time_double_scattering)

    F0_np_list = np.array(F0_append_list)
    max_emission_energy_vs_F0_np = np.array(max_emission_energy_vs_F0)
    F0_np_list_single_scattering = np.array(F0_append_list_sngle_scattering)
    max_emission_energy_vs_F0_np_single_scattering = np.array(max_emission_energy_vs_F0_single_scattering)
    F0_np_list_double_scattering = np.array(F0_append_list_double_scattering)
    max_emission_energy_vs_F0_np_double_scattering = np.array(max_emission_energy_vs_F0_double_scattering)


    plt.plot(F0_np_list*ev/a_b, max_emission_energy_vs_F0_np*ev)
    plt.plot(F0_np_list_single_scattering*ev/a_b, max_emission_energy_vs_F0_np_single_scattering*ev)
    plt.plot(F0_np_list_double_scattering*ev/a_b, max_emission_energy_vs_F0_np_double_scattering*ev)
    plt.xlabel('Field Strength (V/AA)')
    plt.ylabel('Max Emission Energy (eV)')
    plt.title('Max Emission Energy vs Field Strength')
    plt.savefig('fig_max_emission_energy_vs_F0.jpeg')


test = field_strength_scan()

#        print(f'Max emission energy among all time: {max_emission_energy_among_all_time*ev} eV')

#    
#
#    time_list_np = np.array(time_list)
#    max_emission_energy_vs_time = np.array(max_emission_energy_list)
#
#    plt.plot(time_list_np*fs, max_emission_energy_vs_time*ev)
#    plt.xlabel('Time (fs)')
#    plt.ylabel('Max Emission Energy (eV)')
#    plt.title('Max Emission Energy vs Time')
#    plt.savefig('fig_max_emission_energy_vs_time.jpeg')
#
