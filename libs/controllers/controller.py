import numpy as np
import casadi as ca
from libs.utils.normalise_angle import normalise_angle
import matplotlib.pyplot as plt
from libs.vehicle_model.vehicle_model import VehicleParameters

params = VehicleParameters()

class StanleyController:

    def __init__(self, control_gain=2.5, softening_gain=1.0, yaw_rate_gain=0.0, steering_damp_gain=0.0,
                 max_steer=np.deg2rad(24), wheelbase=0.0,
                 waypoints=None):
        """
        Stanley Controller

        At initialisation
        :param control_gain:                (float) time constant [1/s]
        :param softening_gain:              (float) softening gain [m/s]
        :param yaw_rate_gain:               (float) yaw rate gain [rad]
        :param steering_damp_gain:          (float) steering damp gain
        :param max_steer:                   (float) vehicle's steering limits [rad]
        :param wheelbase:                   (float) vehicle's wheelbase [m]
        :param path_x:                      (numpy.ndarray) list of x-coordinates along the path
        :param path_y:                      (numpy.ndarray) list of y-coordinates along the path
        :param path_yaw:                    (numpy.ndarray) list of discrete yaw values along the path
        :param dt:                          (float) discrete time period [s]

        At every time step
        :param x:                           (float) vehicle's x-coordinate [m]
        :param y:                           (float) vehicle's y-coordinate [m]
        :param yaw:                         (float) vehicle's heading [rad]
        :param target_velocity:             (float) vehicle's velocity [m/s]
        :param steering_angle:              (float) vehicle's steering angle [rad]

        :return limited_steering_angle:     (float) steering angle after imposing steering limits [rad]
        :return target_index:               (int) closest path index
        :return crosstrack_error:           (float) distance from closest path index [m]
        """

        self.k = control_gain
        self.k_soft = softening_gain
        self.k_yaw_rate = yaw_rate_gain
        self.k_damp_steer = steering_damp_gain
        self.max_steer = max_steer
        self.L = wheelbase

        self._waypoints = waypoints
        self._lookahead_distance = 5.0
        self.cross_track_deadband = 0.01

        self.px = waypoints[0][:]
        self.py = waypoints[1][:]
        self.pyaw = waypoints[2][:]

    def update_waypoints(self):
        local_waypoints = self._waypoints

    @staticmethod
    def find_target_path_id(px, py, x, y, yaw, param):
        L = param.L

        # Calculate position of the front axle
        fx = x + L * np.cos(yaw)
        fy = y + L * np.sin(yaw)

        dx = fx - px  # Find the x-axis of the front axle relative to the path
        dy = fy - py  # Find the y-axis of the front axle relative to the path

        d = np.hypot(dx, dy)  # Find the distance from the front axle to the path
        target_index = np.argmin(d)  # Find the shortest distance in the array

        return target_index, dx[target_index], dy[target_index], d[target_index]

    def stanley_control(self, x, y, yaw, current_velocity):
        """
        :param x:
        :param y:
        :param yaw:
        :param current_velocity:
        :return: steering output, target index, crosstrack error
        """
        target_index, dx, dy, absolute_error = self.find_target_path_id(self.px, self.py, x, y, yaw, params)
        yaw_error = normalise_angle(self.pyaw[target_index] - yaw)
        # calculate cross-track error
        front_axle_vector = [np.sin(yaw), -np.cos(yaw)]
        nearest_path_vector = [dx, dy]
        crosstrack_error = np.sign(np.dot(nearest_path_vector, front_axle_vector)) * absolute_error
        crosstrack_steering_error = np.arctan2((self.k * crosstrack_error), (self.k_soft + current_velocity))

        desired_steering_angle = yaw_error + crosstrack_steering_error
        # Constrains steering angle to the vehicle limits
        limited_steering_angle = np.clip(desired_steering_angle, -self.max_steer, self.max_steer)

        return limited_steering_angle, target_index, crosstrack_error

    @property
    def waypoints(self):
        return self._waypoints


class MPCC:
    def __init__(self, N, T, param, px, py, pyaw):
        self.N = N  # dt = T/N
        self.T = T  # Total time
        self.params = param  # system parameters
        self.px = px
        self.py = py
        self.pyaw = pyaw
        # casadi setup
        self.nx = 6
        self.nu = 3
        self.controller_cost(0, 0, 0)

    def SystemModel(self, states, u, p, xs=None, us=None):
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

    def controller_cost(self, x, y, yaw):
        """prepares the MPC variables and calculates the cost"""

        # Calculating the contouring cost
        target_index, dx, dy, absolute_error = StanleyController.find_target_path_id(self.px, self.py, x, y, yaw,
                                                                                     self.params)
        selected_px = self.px[target_index:target_index+200]
        selected_py = self.py[target_index:target_index+200]
        selected_pyaw = self.pyaw[target_index:target_index+200]

        e_c = np.sin(selected_pyaw) * (x-selected_px) - np.cos(selected_pyaw) * (y-selected_py)
        e_l = -np.cos(selected_pyaw) * (x - selected_px) - np.sin(selected_pyaw) * (y - selected_py)

        qc = 0.01 # lateral error cost
        ql = 0.005  # longitudinal error cost
        Q_path = np.array([[qc, 0], [0, ql]])

        # Contouring cost
        Jc = 0.
        for k in range(len(e_c)):
            Jc += np.array([e_c[k], e_l[k]]).T @ Q_path @ np.array([e_c[k], e_l[k]])

        #     # unpacking the real signals that come from the system
        #     x_r, y_r, yaw_r, vx_r, vy_r, yaw_dot_r = x_real
        dt = self.T / self.N
        # CasADi works with symbolics
        nx = self.nx
        nu = self.nu

        t = ca.SX.sym("t", 1, 1)
        x = ca.SX.sym("x", nx, 1)
        u = ca.SX.sym("u", nu, 1)
        ode = ca.vertcat(*self.SystemModel(x, u, self.params))
        f = {'x': x, 't': t, 'p': u, 'ode': ode}
        Phi = ca.integrator("Phi", "cvodes", f, {'tf': dt})
        # # Define the decision variable and constraints
        q = ca.vertcat(*[ca.MX.sym(f'u{i}', nu, 1) for i in range(self.N)])
        s = ca.vertcat(*[ca.MX.sym(f'x{i}', nx, 1) for i in range(self.N + 1)])
        # decision variable
        z = []
        # decision variable, lower and upper bounds
        zlb = []
        zub = []
        constraints = []
        # Create a function
        cost = 0.
        Q = np.eye(nx) * 3.6
        R = np.eye(nu) * 0.02

        # Lower bound and upper bound on input
        ulb = [0, 0, 0]
        uub = [10., 10., 10.]

        for i in range(self.N):
            # states
            s_i = s[nx * i:nx * (i + 1)]
            s_ip1 = s[nx * (i + 1):nx * (i + 2)]
            # inputs
            q_i = q[nu * i:nu * (i + 1)]

            # Decision variable
            zlb += [-np.inf] * nx
            zub += [np.inf] * nx
            zlb += ulb
            zub += uub

            z.append(s_i)
            z.append(q_i)

            xt_ip1 = Phi(x0=s_i, p=q_i)['xf']
            cost += s_i.T @ Q @ s_i + q_i.T @ R @ q_i
            constraints.append(xt_ip1 - s_ip1)

        # s_N
        z.append(s_ip1)
        zlb += [-np.inf] * nx
        zub += [np.inf] * nx

        constraints = ca.vertcat(*constraints)
        variables = ca.vertcat(*z)

        # Create the optimization problem
        g_bnd = np.zeros(self.N * nx)
        nlp = {'f': cost, 'g': constraints, 'x': variables}
        opt = {'print_time': 0, 'ipopt.print_level': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opt)
        return solver, zlb, zub

    def solve_mpc(self, current_state):
        """Solve MPC provided the current state, i.e., this
        function is u = h(x), which is the implicit control law of MPC.

        Args:
            current_state (array-like): current state

        Returns:
            tuple: current input and return status pair
        """
        solver, zlb, zub = self.controller_prep()
        g_bnd = np.zeros(self.N * self.nx)

        # Set the lower and upper bound of the decision variable
        # such that s0 = current_state
        for i in range(4):
            zlb[i] = current_state[i]
            zub[i] = current_state[i]
        sol_out = solver(lbx=zlb, ubx=zub, lbg=g_bnd, ubg=g_bnd)
        return np.array(sol_out['x'][self.nx:self.nx + self.nu]), solver.stats()['return_status']


class LongitudinalController:
    def __init__(self, p_gain=1, integral_gain=0, derivative_gain=0):
        self.kp = p_gain
        self.ki = integral_gain
        self.kd = derivative_gain

    def long_control(self, desired_velocity, current_velocity, prev_velocity, v_total_error, dt):
        """
        Longitudinal controller using a simple PID control
        :param desired_velocity: The target velocity that we want to follow
        :param current_velocity: current forward velocity of the vehicle
        :param prev_velocity: previous forward velocity of the vehicle
        :param v_total_error:
        :param dt:
        :return:
        """

        vel_error = desired_velocity - current_velocity
        v_total_error_new = v_total_error + vel_error * dt
        p = self.kp * vel_error
        i = self.ki * v_total_error_new
        d = self.kd * (current_velocity - prev_velocity) / dt
        tau = p + i + d

        if current_velocity <= 0.01:
            tau = abs(tau)

        return v_total_error_new, [tau, tau, tau, tau]


def main():
    print("This script is not meant to be executable, and should be used as a library.")


if __name__ == "__main__":
    main()
