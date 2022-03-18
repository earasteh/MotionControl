import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from libs.utils.env import straight

"""
This file cleans and labels the data and then plots all the results
"""
world = straight


# Variables to track and plot:
# U, V, wz, wFL, wFR, wRL, wRR, yaw, x, y = self.state
# U_dot, V_dot, wz_dot, wFL_dot, wFR_dot, wRL_dot, wRR_dot, yaw_dot, x_dot, y_dot = state_dot
# fFLx, fFRx, fRLx, fRRx, fFLy, fFRy, fRLy, fRRy, fFLz, fFRz, fRLz, fRRz, sFL, sFR, sRL, sRR, fFLxt, fFLyt = outputs

def data_cleaning(DataLog):
    DataLog_nz = DataLog[~np.all(DataLog == 0, axis=1)]  # only plot the rows that are not zeros
    DataLog_pd = pd.DataFrame(DataLog_nz, columns=['time',
                                                   'U', 'V', 'wz', 'wFL', 'wFR', 'wRL', 'wRR', 'yaw', 'x', 'y',
                                                   'U_dot', 'V_dot', 'wz_dot', 'wFL_dot', 'wFR_dot', 'wRL_dot',
                                                   'wRR_dot', 'yaw_dot', 'x_dot', 'y_dot',
                                                   'delta', 'tau_FL', 'tau_FR', 'tau_RL', 'tau_RR',
                                                   'Fx_FL', 'Fx_FR', 'Fx_RL', 'Fx_RR',
                                                   'Fy_FL', 'Fy_FR', 'Fy_RL', 'Fy_RR',
                                                   'Fz_FL', 'Fz_FR', 'Fz_RL', 'Fz_RR',
                                                   'sFL', 'sFR', 'sRL', 'sRR', 'Fxt_FL', 'Fyt_FL', 'crosstrack'])
    if not os.path.exists('results'):
        os.makedirs('results')
    try:
        DataLog_pd.to_csv('results/Results.csv')
    except FileNotFoundError:
        fp = open('results/Results.csv', 'x')
        fp.close()
        DataLog_pd.to_csv('results/Results.csv')

    return DataLog_pd


def plot_results(DataLog_pd1, DataLog_pd2=None, DataLog_pd3=None, figsize_input=None):
    plt.figure(figsize=figsize_input)
    fig_name = 'Forward Velocity'
    plt.title(fig_name)
    plt.plot(DataLog_pd1['time'], DataLog_pd1['U'])
    plt.plot(DataLog_pd2['time'], DataLog_pd2['U'])
    plt.plot(DataLog_pd3['time'], DataLog_pd3['U'])
    plt.grid()
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Forward Velocity (m/s)')
    plt.legend(['T = 0.1', 'T = 0.05', 'T = 0.02'])
    plt.savefig('results/' + fig_name + '.png', dpi=150)

    plt.figure(figsize=figsize_input)
    fig_name = 'Lateral Velocity'
    plt.title(fig_name)
    plt.plot(DataLog_pd1['time'], DataLog_pd1['V'])
    plt.grid()
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Lateral Velocity (m/s)')
    plt.savefig('results/' + fig_name + '.png', dpi=150)

    plt.figure(figsize=figsize_input)
    fig_name = 'Yaw rate'
    plt.title(fig_name)
    plt.plot(DataLog_pd1['time'], DataLog_pd1['yaw_dot'])
    plt.grid()
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Yaw rate (rad/sec)')
    plt.savefig('results/' + fig_name + '.png', dpi=150)

    plt.figure(figsize=figsize_input)
    fig_name = 'Lateral Acceleration'
    plt.title(fig_name)
    plt.plot(DataLog_pd1['time'], DataLog_pd1['V_dot'])
    plt.grid()
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Lateral Acceleration (m/s^2)')
    plt.savefig('results/' + fig_name + '.png', dpi=150)

    # plt.figure()
    # plt.title('Longitudinal Tire Forces')
    # plt.plot(DataLog_pd['time'], DataLog_pd['Fx_FL'])
    # plt.plot(DataLog_pd['time'], DataLog_pd['Fx_FR'])
    # plt.plot(DataLog_pd['time'], DataLog_pd['Fx_RL'])
    # plt.plot(DataLog_pd['time'], DataLog_pd['Fx_RR'])
    # plt.legend(['FL', 'FR', 'RL', 'RR'])
    # plt.xlabel('Time (Sec.)')
    # plt.ylabel('Force (N)')

    plt.figure(figsize=figsize_input)
    fig_name = 'Lateral Tire Forces'
    plt.title(fig_name)
    plt.plot(DataLog_pd1['time'], DataLog_pd1['Fy_FL'])
    plt.plot(DataLog_pd1['time'], DataLog_pd1['Fy_FR'])
    plt.plot(DataLog_pd1['time'], DataLog_pd1['Fy_RL'])
    plt.plot(DataLog_pd1['time'], DataLog_pd1['Fy_RR'])
    plt.grid()
    plt.legend(['FL', 'FR', 'RL', 'RR'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Force (N)')
    plt.savefig('results/' + fig_name + '.png', dpi=150)

    plt.figure(figsize=figsize_input)
    fig_name = 'Normal Tire Forces'
    plt.title(fig_name)
    plt.plot(DataLog_pd1['time'], DataLog_pd1['Fz_FL'])
    plt.plot(DataLog_pd1['time'], DataLog_pd1['Fz_FR'])
    plt.plot(DataLog_pd1['time'], DataLog_pd1['Fz_RL'])
    plt.plot(DataLog_pd1['time'], DataLog_pd1['Fz_RR'])
    plt.grid()
    plt.legend(['FL', 'FR', 'RL', 'RR'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Force (N)')
    plt.savefig('results/' + fig_name + '.png', dpi=150)

    plt.figure()
    plt.title('Torque at each wheel')
    plt.subplot(2, 2, 1)
    plt.plot(DataLog_pd1['time'], DataLog_pd1['tau_FL'])
    plt.legend(['FL'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Wheel Torque (N.m)')
    plt.subplot(2, 2, 2)
    plt.plot(DataLog_pd1['time'], DataLog_pd1['tau_FR'])
    plt.legend(['FR'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Wheel Torque (N.m)')
    plt.subplot(2, 2, 3)
    plt.plot(DataLog_pd1['time'], DataLog_pd1['tau_RL'])
    plt.legend(['RL'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Wheel Torque (N.m)')
    plt.subplot(2, 2, 4)
    plt.plot(DataLog_pd1['time'], DataLog_pd1['tau_RR'])
    plt.legend(['RR'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Wheel Torque (N.m)')
    #
    # plt.figure()
    # plt.title('Angular velocity at each wheel')
    # plt.subplot(2, 2, 1)
    # plt.plot(DataLog_pd['time'], DataLog_pd['wFL'])
    # plt.legend(['FL'])
    # plt.xlabel('Time (Sec.)')
    # plt.ylabel('Angular Vel. (rad/Sec)')
    # plt.subplot(2, 2, 2)
    # plt.plot(DataLog_pd['time'], DataLog_pd['wFR'])
    # plt.legend(['FR'])
    # plt.xlabel('Time (Sec.)')
    # plt.ylabel('Angular Vel. (rad/Sec)')
    # plt.subplot(2, 2, 3)
    # plt.plot(DataLog_pd['time'], DataLog_pd['wRL'])
    # plt.legend(['RL'])
    # plt.xlabel('Time (Sec.)')
    # plt.ylabel('Angular Vel. (rad/Sec)')
    # plt.subplot(2, 2, 4)
    # plt.plot(DataLog_pd['time'], DataLog_pd['wRR'])
    # plt.legend(['RR'])
    # plt.xlabel('Time (Sec.)')
    # plt.ylabel('Angular Vel. (rad/Sec)')

    plt.figure(figsize=figsize_input)
    fig_name = 'Steering Angle'
    plt.title(fig_name)
    plt.plot(DataLog_pd1['time'], DataLog_pd1['delta'] * 180 / np.pi)
    if DataLog_pd2 is not None:
        plt.plot(DataLog_pd2['time'], DataLog_pd2['delta'] * 180 / np.pi)
    if DataLog_pd3 is not None:
        plt.plot(DataLog_pd3['time'], DataLog_pd3['delta'] * 180 / np.pi)
    plt.grid()
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Steering angle (deg)')
    plt.legend(['T = 0.1', 'T = 0.05', 'T = 0.02'])
    plt.savefig('results/' + fig_name + '.png', dpi=150)

    # fig, ax = plt.subplots(figsize=figsize_input)
    # ax.set_xlim(0.1, 5)
    # fig_name = 'Cross Track Error'
    # plt.title(fig_name)
    # plt.plot(DataLog_pd1['time'], DataLog_pd1['crosstrack'])
    # plt.grid()
    # plt.xlabel('Time (Sec.)')
    # plt.ylabel('Cross Track Error (m)')
    # plt.savefig('results/' + fig_name + '.png', dpi=150)

    plt.figure(figsize=figsize_input)
    fig_name = 'Combined slip'
    plt.title(fig_name)
    plt.subplot(2, 2, 1)
    plt.plot(DataLog_pd1['time'], DataLog_pd1['sFL'])
    plt.grid()
    plt.legend(['FL'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Slip')
    plt.subplot(2, 2, 2)
    plt.plot(DataLog_pd1['time'], DataLog_pd1['sFR'])
    plt.grid()
    plt.legend(['FR'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Slip')
    plt.subplot(2, 2, 3)
    plt.plot(DataLog_pd1['time'], DataLog_pd1['sRL'])
    plt.grid()
    plt.legend(['RL'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Slip')
    plt.subplot(2, 2, 4)
    plt.plot(DataLog_pd1['time'], DataLog_pd1['sRR'])
    plt.grid()
    plt.legend(['RR'])
    plt.xlabel('Time (Sec.)')
    plt.ylabel('Slip')
    plt.savefig('results/' + fig_name + '.png', dpi=150)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.grid()
    fig_name = 'Trajectory of the vehicle'
    plt.title(fig_name)
    plt.plot(DataLog_pd1['x'], DataLog_pd1['y'])
    plt.plot(DataLog_pd2['x'], DataLog_pd2['y'])
    plt.plot(DataLog_pd3['x'], DataLog_pd3['y'])
    plt.plot(world.xm, world.ym, 'k--')
    plt.fill(world.obstacle_x, world.obstacle_y, color='red', zorder=2)
    # ax.set_xlim(-10, 100)
    # ax.set_ylim(-5, 25)
    # ax.set_aspect('equal')
    ax.legend(['Generated Path', 'Ref'])
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend(['T = 0.1', 'T = 0.05', 'T = 0.02'])
    plt.savefig('results/' + fig_name + '.png', dpi=150)

    plt.show()


def main():
    curr_path = os.getcwd()
    os.chdir('../..')
    DataLog_pd10 = pd.read_csv('results/10/Results.csv')
    DataLog_pd5 = pd.read_csv('results/5/Results.csv')
    DataLog_pd2 = pd.read_csv('results/2/Results.csv')
    plot_results(DataLog_pd10, DataLog_pd5, DataLog_pd2)


if __name__ == "__main__":
    main()
