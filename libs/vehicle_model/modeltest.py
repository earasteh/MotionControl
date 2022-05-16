import numpy as np
import matplotlib.pyplot as plt

# Bf1 = 20.6357
# Cf1 = 1.5047
# Df1 = 1.1233
#
# Bf2 = 2.579
# Cf2 = 1.2
# Df2 = 0.192
#
#
# alpha_f = np.linspace(-20, 20, 1000) * np.pi / 180
#
# Ffy_1 = Df1 * np.sin(Cf1 * np.arctan(Bf1 * alpha_f))
# Ffy_2 = Df2 * np.sin(Cf2 * np.arctan(Bf2 * alpha_f))
#
# plt.figure()
# plt.plot(alpha_f * 180/np.pi, Ffy_1)
# plt.plot(alpha_f * 180/np.pi, Ffy_2)
# plt.xlabel(r'$\alpha (degrees)$')
# plt.ylabel(r'$F_y (N)$')
# plt.legend(['Normal values', 'MPCC values'])
# plt.grid()
# plt.show()


from libs.controllers.controller import MPC
from libs.vehicle_model.vehicle_model import VehicleParameters
from acados_template import AcadosSim, AcadosSimSolver

parameters = VehicleParameters()

px = np.linspace(0, 100, 1000)
py = np.zeros((1, 1000))
pyaw = np.zeros((1, 1000))
init = np.array([0, 0, 0, 10, 0, 0, 0, 2*np.pi/180])
N = 20
Tf = 0.1

test_mpc = MPC(N, Tf, param=parameters, px=px, py=py, pyaw=pyaw, veh_initial_conditions=init)

sim = AcadosSim()
model, constraint = test_mpc.bicycle_model(init, parameters)
sim.model = model

nx = model.x.size()[0]
nu = model.u.size()[0]

# set simulation time
sim.solver_options.T = Tf
# set options
sim.solver_options.integrator_type = 'IRK'
sim.solver_options.num_stages = 3
sim.solver_options.num_steps = 3
sim.solver_options.newton_iter = 3  # for implicit integrator
sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"

# create
acados_integrator = AcadosSimSolver(sim)

simX = np.ndarray((N + 1, nx))
x0 = init
u0 = np.array([0.0, 0.0])
acados_integrator.set("u", u0)

simX[0, :] = x0

for i in range(N):
    if simX[i, 7] < 2 * np.pi / 180:
        acados_integrator.set("u", np.array([0.0, 0.1 * np.pi / 180]))

    # set initial state
    acados_integrator.set("x", simX[i, :])
    # initialize IRK
    if sim.solver_options.integrator_type == 'ERK':
        acados_integrator.set("xdot", np.zeros((nx,)))

    # solve
    status = acados_integrator.solve()
    # get solution
    simX[i + 1, :] = acados_integrator.get("x")

if status != 0:
    raise Exception('acados returned status {}. Exiting.'.format(status))

S_forw = acados_integrator.get("S_forw")
print("S_forw, sensitivities of simulaition result wrt x,u:\n", S_forw)

plt.figure()
plt.plot(simX[:, 0], simX[:, 1])
plt.axis('equal')
plt.xlabel('X')
plt.ylabel('Y')

t = np.linspace(0, Tf, N+1)

plt.figure()
plt.plot(t, simX[:, 7] * 180/np.pi)
plt.plot(t, simX[:, 3])
plt.plot(t, simX[:, 4])
plt.plot(t, simX[:, 5])
plt.legend(['delta', 'Vx', 'Vy', 'yaw rate'])
plt.show()
