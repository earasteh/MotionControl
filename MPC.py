from scipy.optimize import fsolve
from casadi import *
import matplotlib.pyplot as plt
import numpy as np
from SystemModel import VehicleModel, param


# model used for MPC propagation
def model(states, u, p, xs=None, us=None):
    """Compute the right-hand side of the ODEs

    Args:
        states (array-like): State vector
        u (array-like): Input vector
        p (object of class parameter): Parameters
        xs (array-like, optional): steady-state
        us (array-like, optional): steady-state input

    Returns:
        array-like: dx/dt
    """

    if xs is not None:
        # Assume x is in deviation variable form
        states = [states[i] + xs[i] for i in range(6)]

    if us is not None:
        # Assume u is in deviation variable form
        u = [u[i] + us[i] for i in range(2)]

    # parameters:
    m = p.m
    lf = p.a
    lr = p.b

    # x, y, yaw, vx, vy, yaw_dot = states
    # tau, ddelta, delta = u
    x = states[0]
    y = states[1]
    yaw = states[2]
    vx = states[3]
    vy = states[4]
    yaw_dot = states[5]

    tau = u[0]
    ddelta = u[1]
    delta = u[2]

    # TODO: Fix this
    Fx = 10 * tau

    dx = vx * np.cos(yaw) - vy * np.sin(yaw)
    dy = vx * np.sin(yaw) + vy * np.cos(yaw)
    dyaw = yaw_dot
    dvx = Fx / m
    dvy = (ddelta * vx + delta * dvx) * lr / (lf + lr)
    dyaw_dot = (ddelta * vx + delta * dvx) * 1 / (lf + lr)

    state_dot = [dx, dy, dyaw, dvx, dvy, dyaw_dot]

    return state_dot


# Get the steady-state
# us = np.array([1, 2.4310])
# sigma = [1.0, 0.4]
# f = lambda x: model(x, us, sigma)
# xs, _, flag, _ = fsolve(f, np.array([0.4, 1.6, 0.38, 0.24]), full_output=True)

# print(f'xs = {xs}')
# print(f'Exit flag: {flag}')

# Parameters (used in optimal control problem later as well)
T = 10.0
N = 100
dt = T / N

# CasADi works with symbolics
t = SX.sym("t", 1, 1)
x = SX.sym("x", 6, 1)
u = SX.sym("u", 3, 1)
ode = vertcat(*model(x, u, param))
print(ode, ode.shape)

f = {'x': x, 't': t, 'p': u, 'ode': ode}
Phi = integrator("Phi", "cvodes", f, {'tf': dt})
#
# System Model - in general, the system model and the MPC model are different
U_init = 8.33
state0 = [U_init, 0, 0,  # U, V, wz
          U_init / param.rw, U_init / param.rw, U_init / param.rw, U_init / param.rw,  # wFL, wFR, wRL, wRR
          0, 0, 0]  # yaw, x, y
tire_torque0 = [0, 0, 0, 0]
delta0 = [0, 0, 0, 0]
mu_max = [1, 1, 1, 1]
ax_prev0 = 0
ay_prev0 = 0

veh = VehicleModel()
system = veh.planar_model(state0, tire_torque0, mu_max, delta0, param, ax_prev0, ay_prev0)

# # Define the decision variable and constraints
# q = vertcat(*[MX.sym(f'u{i}', 2, 1) for i in range(N)])
# s = vertcat(*[MX.sym(f'x{i}', 4, 1) for i in range(N + 1)])
# # decision variable
# z = []
# # decision variable, lower and upper bounds
# zlb = []
# zub = []
# constraints = []
#
# # Create a function
# cost = 0.
# Q = np.eye(4) * 3.6
# R = np.eye(2) * 0.02
#
# # Lower bound and upper bound on input
# ulb = list(-us)
# uub = list(np.array([10., 15.]) - us)
#
# for i in range(N):
#     # states
#     s_i = s[4 * i:4 * (i + 1)]
#     s_ip1 = s[4 * (i + 1):4 * (i + 2)]
#     # inputs
#     q_i = q[2 * i:2 * (i + 1)]
#
#     # Decision variable
#     zlb += [-np.inf] * 4
#     zub += [np.inf] * 4
#     zlb += ulb
#     zub += uub
#
#     z.append(s_i)
#     z.append(q_i)
#
#     xt_ip1 = Phi(x0=s_i, p=q_i)['xf']
#     cost += s_i.T @ Q @ s_i + q_i.T @ R @ q_i
#     constraints.append(xt_ip1 - s_ip1)
#
# # s_N
# z.append(s_ip1)
# zlb += [-np.inf] * 4
# zub += [np.inf] * 4
#
# constraints = vertcat(*constraints)
# variables = vertcat(*z)
#
# # Create the optmization problem
# g_bnd = np.zeros(N * 4)
# nlp = {'f': cost, 'g': constraints, 'x': variables}
# opt = {'print_time': 0, 'ipopt.print_level': 0}
# solver = nlpsol('solver', 'ipopt', nlp, opt)
#
#
# def solve_mpc(current_state):
#     """Solve MPC provided the current state, i.e., this
#     function is u = h(x), which is the implicit control law of MPC.
#
#     Args:
#         current_state (array-like): current state
#
#     Returns:
#         tuple: current input and return status pair
#     """
#
#     # Set the lower and upper bound of the decision variable
#     # such that s0 = current_state
#     for i in range(4):
#         zlb[i] = current_state[i]
#         zub[i] = current_state[i]
#     sol_out = solver(lbx=zlb, ubx=zub, lbg=g_bnd, ubg=g_bnd)
#     return (np.array(sol_out['x'][4:6]), solver.stats()['return_status'])
#
#
# ## Main Simulation Loop
# num_time_steps_sim = 20  # number of time steps in simulation
#
# # Store the system states and control actions applied to the system
# # in array
# state_history = np.zeros((num_time_steps_sim + 1, 4))
# input_history = np.zeros((num_time_steps_sim + 1, 2))
#
# # Set current state - using deviation variables
# state_history[0, :] = np.array([0.5, 0.0, 0.7, 0.7]) - xs
# current_state = state_history[0, :]
#
# # Time array for plotting
# time = [i * dt for i in range(num_time_steps_sim + 1)]
#
# # Closed-loop simulation
# for k in range(num_time_steps_sim):
#     print(f'Current time: {k * dt}')
#     current_control, status = solve_mpc(current_state)
#     print(f'Solver status: {status}')
#
#     # Advance the simulation one time step
#     # Set current_state to be the state at the next time steps
#     current_state = np.array(system(x0=current_state, p=current_control)['xf'])
#
#     current_state = current_state.reshape((4,))
#     current_control = current_control.reshape((2,))
#
#     # Save data for plotting
#     input_history[k, :] = current_control
#     state_history[k + 1:k + 2, :] = current_state
#
# # Save the last control one more time for plotting
# input_history[-1, :] = current_control
#
# # Figure
# plt.figure(figsize=[14, 14])
# fig, axs = plt.subplots(6, 1, figsize=[10, 10])
# t_max = min(10, num_time_steps_sim * dt)
# for j in range(4):
#     axs[j].plot(time, state_history[:, j] + xs[j], 'k-',
#                 [time[0], time[-1]], [xs[j], xs[j]], 'k--')
#     axs[j].set_ylabel(f'$x_{j + 1}$')
#     axs[j].set_xlim([0, t_max])
#
# for j in range(2):
#     axs[j + 4].step(time, input_history[:, j] + us[j], 'k-',
#                     [time[0], time[-1]], [us[j], us[j]], 'k--',
#                     where='post')
#     axs[j + 4].set_ylabel(f'$u_{j + 1}$')
#     axs[j + 4].set_xlim([0, t_max])
# axs[-1].set_xlabel('Time')
#
# plt.show()
