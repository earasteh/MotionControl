import numpy as np
import matplotlib.pyplot as plt

Bf1 = 20.6357
Cf1 = 1.5047
Df1 = 1.1233

Bf2 = 2.579
Cf2 = 1.2
Df2 = 0.192


m = 3600 * 0.453592
g = 9.81
l = 0.5

alpha_f = np.linspace(-20, 20, 1000) * np.pi / 180

Ffy_1 = Df1 * np.sin(Cf1 * np.arctan(Bf1 * alpha_f)) * 500
Ffy_2 = Df1 * np.sin(Cf1 * np.arctan(Bf1 * alpha_f)) * 1000
Ffy_3 = Df1 * np.sin(Cf1 * np.arctan(Bf1 * alpha_f)) * 2000

plt.figure()
plt.plot(alpha_f * 180/np.pi, Ffy_1)
plt.plot(alpha_f * 180/np.pi, Ffy_2)
plt.plot(alpha_f * 180/np.pi, Ffy_3)
plt.xlabel(r'$\alpha_f (degrees)$')
plt.ylabel(r'$F_y (N)$')
plt.legend(['Normal Force = 500 N', 'Normal Force = 1000 N', 'Normal Force = 2000 N'])
plt.grid()
plt.show()
