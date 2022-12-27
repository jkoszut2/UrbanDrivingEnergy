import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp


m = 1611+82             # average vehicle mass + driver [kg]
rho = 1.225             # air density [kg/m^3]
Cd = 0.23               # drag coefficient
Aref = 2.22             # average vehicle frontal area [m^2]
g = 9.81                # gravity constant [m/s^2]
Crr = 0.01              # rolling resistance coefficient
Cv = 0.3                # kinetic friction coefficient
r = 0.334               # wheel radius [m] (235/45R18 tire)

capacity_kWh = 54       # [kWh]
Voc = 360               # [V]
Pmax_kW = 211           # [kW]
Tmax_Nm = 375           # [Nm]
Tmax_rpm = Pmax_kW*1000/Tmax_Nm*60/2/np.pi # motor speed at max torque [rpm]
gear_ratio = 9          # motor to driven axle gear ratio
max_speed = 16000       # maximum motor speed [rpm]
dt_eff = 0.85           # drivetrain efficiency
Rcirc = 0.05            # battery internal resistance [ohms]


fig, axs = plt.subplots(3,2, figsize=(12,8), sharex=False)
axs = axs.flatten()
legend_font = 8
title = 'Dependency of Energy Consumption on Driving Style'
legend_elems = []
fig.suptitle(title)

for test in range(3):
    # Define optimization vars
    if test == 0:
        N = 185
    else:
        N = 144
    dt = 1
    M = 1e4
    x1 = cp.Variable(N+1) # Position
    x2 = cp.Variable(N+1) # Velocity
    T = cp.Variable(N) # Powertrain torque
    Fdrag = cp.Variable(N) # Vehicle drag
    E = cp.Variable(N+1) # Battery energy
    Pmot = cp.Variable(N) # Battery power
    Vcirc = cp.Variable(N) # Circuit voltage
    I = cp.Variable(N) # Motor current
    SOC = cp.Variable(N+1) # Battery SOC
    pedal = cp.Variable(N, boolean=True)

    # Define objective function
    if test == 0:
        title = 'Minimum Energy' # Minimum energy
        objective = cp.Minimize( E[0]-E[N] )
    elif test == 1:
        # Minimum time - Lower N until solver fails
        title = 'Minimum Time' # Minimum energy
        objective = cp.Minimize( E[0]-E[N] )
    else:
        # Minimum time - Lower N until solver fails
        title = 'Maximum Velocity' # Maximum velocity
        objective = cp.Minimize( E[0]-E[N] - cp.sum(x1)/10 )

    # Define constraints
    pos_final = 1700 # m
    vel_lower = 0 # m/s
    vel_upper = 16 # m/s
    acc_abs = 1.5 # m/s^2
    jerk_abs = 1 # m/s^3
    constraints = [ ]
    constraints += [ x1[0] == 0 ]
    constraints += [ x1[N] == pos_final ]
    # vel_init = 0 m/s
    constraints += [ x2[0] == 0 ]
    # vel_final = 0 m/s
    constraints += [ x2[N] == 0 ]
    # # acc_init = 0 m/s^2
    constraints += [ x2[1] == x2[0] ]
    # # acc_final = 0 m/s^2
    constraints += [ x2[N] == x2[N-1] ]
    # Traffic light constraints
    # Define a traffic light as [initial time, final time, position]
    tls = []
    redlight = 35 # redlight duration [sec]
    greenlight = 35 # greenlight duration [sec]
    block = 500 # length of one block [m]
    for intersection in np.arange(0, pos_final, block):
        for start in np.arange(20, N*dt, redlight + greenlight):
            tls.append([start, start + redlight, intersection])
    tls_bools = {}
    tls_on = True # Control whether or not traffic lights are used
    # The for loop below will automatically create constraints for all traffic lights
    for i in range(len(tls)):
        # Ensure that traffic light fits into optimization horizon
        if int(tls[i][0]/dt) <= N and tls_on:            
            dict_index = "tl{0}".format(i)
            tls_bools[dict_index] = cp.Variable(1, boolean=True)
            # Clip end of traffic light if it goes past optimization horizon
            t_initial = int(tls[i][0]/dt)
            if int(tls[i][1]/dt) > N:
                t_final = N
            else:
                t_final = int(tls[i][1]/dt)
            position = tls[i][2]
            constraints += [ x1[t_final]   - position <=  M*tls_bools[dict_index] ]
            constraints += [ x1[t_initial] - position >= -M*(1-tls_bools[dict_index]) ]

    for k in range(0,N):
        constraints += [ x2[k+1] >= vel_lower ]
        constraints += [ x2[k+1] <= vel_upper ]
        constraints += [ (x2[k+1] - x2[k])/dt >= -acc_abs ]
        constraints += [ (x2[k+1] - x2[k])/dt <= acc_abs ]
        constraints += [ x1[k+1] == x1[k] + x2[k]*dt ]
        constraints += [ x2[k+1] == x2[k] + ( (T[k])/m/r*gear_ratio*dt_eff - Fdrag[k]/m)*dt ]
        constraints += [ Fdrag[k] >= 0.5*rho*Cd*Aref*x2[k]**2 + Cv*x2[k] + Crr*m*g*pedal[k] ]
        constraints += [ E[k+1] == E[k] - Voc*Vcirc[k]/Rcirc*dt/3600 ]
        constraints += [ x2[k]*60/(2*np.pi*r)*gear_ratio <= max_speed ]
        constraints += [ T[k] <= Tmax_Nm/(max_speed-Tmax_rpm)*max_speed \
                                 + (Tmax_Nm/-(max_speed-Tmax_rpm))*x2[k]*60/(2*np.pi*r)*gear_ratio ]
        constraints += [ T[k] >= - Tmax_Nm/(max_speed-Tmax_rpm)*max_speed \
                                 - (Tmax_Nm/-(max_speed-Tmax_rpm))*x2[k]*60/(2*np.pi*r)*gear_ratio ]

    # SOC
    constraints += [ SOC[0] == 0.8 ]
    constraints += [ SOC == E/capacity_kWh/1000]
    constraints += [ SOC <= 0.8 ]
    constraints += [ SOC >= 0.2 ]
    # Power
    # Relaxed SOC constraint --> Becomes tight to minimize energy loss
    constraints += [ Pmot <= Vcirc/Rcirc*Voc - (Vcirc)**2/Rcirc ]
    constraints += [ T <= Tmax_Nm ]
    constraints += [ T >= -Tmax_Nm ]
    constraints += [ T == Pmot/(Pmax_kW*1000)*Tmax_Nm ]
    # Voc
    constraints += [ Vcirc/Rcirc <= (Pmax_kW*1000)/Voc ] # Current limit

    for k in range(1, N):
        # Minimize jerk (derivative of acceleration)
        constraints += [ (x2[k+1] - 2*x2[k] + x2[k-1])/dt**2 <= jerk_abs ]
        constraints += [ (x2[k+1] - 2*x2[k] + x2[k-1])/dt**2 >= -jerk_abs ]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True, solver='ECOS_BB')
    print(prob.status)

    # Calculate recovered energy
    E_regen = np.zeros(N+1)
    for i in range(1,len(E_regen)):
        if Vcirc.value[i-1] < 0:
            E_regen[i] = E_regen[i-1] - Voc*Vcirc.value[i-1]/Rcirc*dt/3600
        else:
            E_regen[i] = E_regen[i-1]

    print(f"======{'='*len(title)}======")
    print(f"===== {title} =====")
    print(f"======{'='*len(title)}======")
    print(f"Min SOC = {np.min(SOC.value):.5}")
    print(f"Final SOC = {(SOC.value[-1]):.5}")
    print(f"Recovered SOC = {E_regen[-1]/(capacity_kWh*1000):.5}")
    print(f"Estimated Range = {1/(SOC.value[0]-SOC.value[-1])*pos_final/1609.34:.5} miles")
    print(f"Time of Travel = {np.where(x1.value>=pos_final-0.00001)[0][0]*dt:.0f} seconds")
    print(f"Max Torque = {np.max(T.value):.2f} Nm")
    print(f"Max Power = {np.max(Pmot.value)/1000:.2f} kW")
    print(f"Max Motor Speed = {np.max(x2.value*60/(2*np.pi*r)*gear_ratio):.0f} RPM")
    print(f"Max Vel = {np.max((x2.value)):.4} m/s")
    print(f"Max Acc = {np.max((x2.value[1:]-x2.value[:-1])/dt):.4} m/s^2")
    print(f"Max Jerk = {np.max((x2.value[2:]-2*x2.value[1:-1]+x2.value[:-2])/dt**2):.4} m/s^3")

    time = np.arange(0, (N+1)*dt, dt)
    # Plot order
    plt_pos = 0
    plt_vel = 2
    plt_acc = 4
    plt_jerk = 5
    plt_power = 3
    plt_soc = 1
    axs[plt_pos].plot(time, x1.value, label='_nolegend_')
    axs[plt_pos].set_ylabel('Position [m]')
    for i in range(len(tls)):
        # Ensure that traffic light fits into optimization horizon
        if int(tls[i][plt_pos]/dt) <= N:
            axs[plt_pos].hlines(y=tls[i][2], xmin=tls[i][0], xmax=tls[i][1],
                                linewidth=2, color='r')

    axs[plt_pos].hlines(y=pos_final, xmin=0, xmax=int(N*dt), color='k',
                        linestyle='--', linewidth=1)
    axs[plt_pos].legend(['Red Lights'], loc='best', fontsize=legend_font)
    axs[plt_acc].plot((time[1:]+time[:-1])/2, (x2.value[1:]-x2.value[:-1])/dt)
    if test == 0:
        axs[plt_acc].hlines(y=-acc_abs, xmin=0, xmax=int(N*dt), color='k',
                            linestyle='--', linewidth=1, label='Limits')
        axs[plt_acc].hlines(y=acc_abs, xmin=0, xmax=int(N*dt), color='k',
                            linestyle='--', linewidth=1)
    axs[plt_acc].legend(loc='upper right', fontsize=legend_font)
    axs[plt_acc].set_ylabel('Acceleration [m/s^2]')
    axs[plt_vel].plot(time, x2.value)
    if test == 0:
        axs[plt_vel].hlines(y=vel_lower, xmin=0, xmax=int(N*dt), color='k',
                            linestyle='--', linewidth=1, label='Limits')
        axs[plt_vel].hlines(y=vel_upper, xmin=0, xmax=int(N*dt), color='k',
                            linestyle='--', linewidth=1)
    axs[plt_vel].legend(loc='upper right', fontsize=legend_font)
    axs[plt_vel].set_ylabel('Velocity [m/s]')
    axs[plt_jerk].plot((time[2:]+time[:-2])/2,
                       (x2.value[2:]-2*x2.value[1:-1]+x2.value[:-2])/dt**2)
    if test == 0:
        axs[plt_jerk].hlines(y=-jerk_abs, xmin=0, xmax=int(N*dt), color='k',
                             linestyle='--', linewidth=1, label='Limits')
        axs[plt_jerk].hlines(y=jerk_abs, xmin=0, xmax=int(N*dt), color='k',
                             linestyle='--', linewidth=1)
    axs[plt_jerk].legend(loc='best', fontsize=legend_font)
    axs[plt_jerk].set_ylabel('Jerk [m/s^3]')
    axs[plt_power].plot(time[:-1], Pmot.value/1000)
    axs[plt_power].set_ylabel('Power [KW]')
    axs[plt_soc].plot(time, SOC.value*100, label='SOC')
    axs[plt_soc].set_ylabel('State of Charge [%]')
    axs[plt_soc].yaxis.set_major_formatter(plt.matplotlib.ticker.\
                 StrMethodFormatter('{x:,.1f}'))
    axs[plt_acc].set_xlabel('Time [s]')
    axs[-1].set_xlabel('Time [s]')
    legend_elems = np.append(legend_elems, title)
axs[plt_soc].legend(legend_elems, loc='best', fontsize=legend_font)
plt.tight_layout()
plt.subplots_adjust(left = 0.0875, right = 0.9125, bottom = 0.0875,
                    top = 0.9125, wspace = 0.25, hspace = 0.2)
plt.show()

