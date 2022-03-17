import numpy as np
import matplotlib.pyplot as plt

Bf1 = 20.6357
Cf1 = 1.5047
Df1 = 1.1233

Bf2 = 2.579
Cf2 = 1.2
Df2 = 0.192


alpha_f = np.linspace(-20, 20, 1000) * np.pi / 180

Ffy_1 = Df1 * np.sin(Cf1 * np.arctan(Bf1 * alpha_f))
Ffy_2 = Df2 * np.sin(Cf2 * np.arctan(Bf2 * alpha_f))

plt.figure()
plt.plot(alpha_f * 180/np.pi, Ffy_1)
plt.plot(alpha_f * 180/np.pi, Ffy_2)
plt.xlabel(r'$\alpha (degrees)$')
plt.ylabel(r'$F_y (N)$')
plt.legend(['Normal values', 'MPCC values'])
plt.grid()
plt.show()
