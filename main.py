import numpy as np
from matplotlib import pyplot as plt

## Parameters
ev = 27.2114
fs = 0.024189
a_b = 0.529177
eps_g = 4.18/ev
mu_eh = 0.083

dt_traj = 1.0
tprop = 120.0/fs

## Laser field parameters
omega = 0.387/ev
#F0 = 0.165*a_b/ev

tau = 9*2*np.pi/omega

dt_ex = tau/20

def A_field_t(t, F0):
    """
    Returns the vector potential of the laser field at time t.
    """

    A0 = F0/omega

    if t < 0:
        return 0.0
    elif t < tau:
        return A0 * np.sin(omega * t) * np.sin(np.pi * t / tau)**4
    else:
        return 0.0

    
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

    k_recomb = []
    while t < tprop:
        k = k0 + A_field_t(t+dt_traj*0.5, F0)
        v = velocity_kane_band(k)
        x_new = x + v*dt_traj
        if(x_new == 0 or x_new*x <0):
            k_recomb.append(k)
        x = x_new
        t += dt_traj

    if len(k_recomb) == 0:
        return -1.0

    k_recomb_np = np.array(k_recomb)
    emission_energy = eps_kane_band(k_recomb_np)
    max_emission_energy = np.max(emission_energy)
    
    return max_emission_energy
            
def run_single_scattered_trajectory_analysis(F0):
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

def field_strength_scan():
    """
    Scans the field strength and analyzes the scattered trajectories.
    """

    F0_VpAA_list = np.linspace(0.01, 0.3, 30)
    F0_list = F0_VpAA_list*a_b/ev


    F0_append_list = []
    max_emission_energy_vs_F0 = []

    for F0 in F0_list:
        max_emission_energy_among_all_time = run_single_scattered_trajectory_analysis(F0)
        if max_emission_energy_among_all_time > 0:
            F0_append_list.append(F0)
            max_emission_energy_vs_F0.append(max_emission_energy_among_all_time)

    F0_np_list = np.array(F0_append_list)
    max_emission_energy_vs_F0_np = np.array(max_emission_energy_vs_F0)

    plt.plot(F0_np_list, max_emission_energy_vs_F0_np*ev)
    plt.xlabel('Field Strength (a.u.)')
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
