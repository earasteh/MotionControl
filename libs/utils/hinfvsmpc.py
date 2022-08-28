import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from numpy import pi

hinf = scipy.io.loadmat('../../../MATLAB/Hinf_MotionControl/results/hinf_results.mat')
youla = scipy.io.loadmat('../../../MATLAB/Hinf_MotionControl/results/youla_results.mat')


DataLog_pd = pd.read_csv('../../results/Results.csv')

plt.figure()
plt.title('MPC vs Robust Control', fontsize=16)
plt.plot(DataLog_pd['x'], DataLog_pd['y'], color='green')
plt.plot(hinf['x_inf'], hinf['y_inf'], color='red')
plt.plot(youla['x_youla'], youla['y_youla'], color='blue')
plt.legend(['MPC', 'Hinf', 'Youla'])
# plt.legend(['MPC', 'Youla'])
plt.grid()
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.savefig('results/' + 'mpcvsYoula_traj' + '.png', dpi=150)

plt.figure()
plt.title('MPC vs Youla', fontsize=16)
plt.plot(DataLog_pd['time'], DataLog_pd['yaw'], color='green')
plt.plot(hinf['t_inf'], hinf['yaw_inf'], color='red')
plt.plot(youla['t_youla'], youla['yaw_youla'], color='blue')
plt.legend(['MPC', 'Hinf', 'Youla'])
# plt.legend(['MPC', 'Youla'])
plt.grid()
plt.xlabel('Time (Sec)')
plt.ylabel('Yaw (rad)')
plt.savefig('results/' + 'mpcvsYoula_yaw' + '.png', dpi=150)

plt.figure()
plt.title('MPC vs Youla', fontsize=16)
plt.plot(DataLog_pd['time'], DataLog_pd['delta'] * 180/pi, color='green')
plt.plot(hinf['t_inf'], hinf['delta_inf'] * 180/pi, color='red')
plt.plot(youla['t_youla'], youla['delta_youla'] * 180/pi, color='blue')
plt.grid()
plt.legend(['MPC', 'Hinf', 'Youla'])
# plt.legend(['MPC', 'Youla'])
plt.xlabel('Time (sec)')
plt.ylabel('delta (deg.)')
plt.savefig('results/' + 'mpcvsYoula_delta' + '.png', dpi=150)
plt.show()


# plot_results(data_cleaning(DataLog_pd), WhichPlots=('traj'))