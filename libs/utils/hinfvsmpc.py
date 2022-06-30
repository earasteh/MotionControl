import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
from numpy import pi

hinf = scipy.io.loadmat('../../../MATLAB/Hinf_MotionControl/results/hinf_results.mat')

# for key, value in hinf.items():
#     print(key)
# plt.figure()
# plt.plot(x_hinf, y_hinf)
# plt.show()


DataLog_pd = pd.read_csv('../../results/Results.csv')

plt.figure()
plt.title('MPC vs Hinf', fontsize=16)
plt.plot(DataLog_pd['x'], DataLog_pd['y']-10, color='green')
plt.plot(hinf['x'], hinf['y'], color='red')
plt.legend(['MPC', 'Hinf'])
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.show()


plt.figure()
plt.title('MPC vs Hinf', fontsize=16)
plt.plot(DataLog_pd['time'], DataLog_pd['delta'] * 180/pi, color='green')
plt.plot(hinf['t'], hinf['delta'] * 180/pi, color='red')
plt.legend(['MPC', 'Hinf'])
plt.xlabel('Time (sec)')
plt.ylabel('delta (deg.)')
plt.show()



# plot_results(data_cleaning(DataLog_pd), WhichPlots=('traj'))