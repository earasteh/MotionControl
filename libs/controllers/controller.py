import numpy as np
import casadi as ca
from libs.utils.normalise_angle import normalise_angle
import matplotlib.pyplot as plt
from libs.vehicle_model.vehicle_model import VehicleParameters


# params = VehicleParameters()


class StanleyController:

    def __init__(self, control_gain=2.5, softening_gain=1.0, yaw_rate_gain=0.0, steering_damp_gain=0.0,
                 max_steer=np.deg2rad(24), wheelbase=0.0, param=None,
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
        self.params = param  # system parameters

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

        return target_index, dx[target_index], dy[target_index], d[target_index], d

    def stanley_control(self, x, y, yaw, current_velocity):
        """
        :param x:
        :param y:
        :param yaw:
        :param current_velocity:
        :return: steering output, target index, crosstrack error
        """
        target_index, dx, dy, absolute_error, _ = self.find_target_path_id(self.px, self.py, x, y, yaw, self.params)
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
        self.nu = 2
        self.controller_cost(0, 0, 0)

    def SystemModel(self, states, u, param, xs=None, us=None):
        """Compute the right-hand side of the ODEs

        Args:
            states (array-like): State vector
            u (array-like): Input vector
            p (object of class parameter): Parameters
            xs (array-like, optional): steady-state
            us (array-like, optional): steady-state input

        Returns:
            array-like: dx/dt
            :param us:
            :param xs:
            :param states:
            :param param:
        """
        # parameters of the vehicle:
        m = param.m
        lf = param.a
        lr = param.b
        Izz = param.Izz
        rw = param.rw
        # tire parameters for the pajecka model - front
        Bf = param.BFL
        Cf = param.CFL
        Df = param.DFL
        # tire parameters for the pajecka model - rear
        Br = param.BRL
        Cr = param.CRL
        Dr = param.DRL
        # Longitudinal parameters
        Cm1 = 0.287
        Cm2 = 0.0545
        Cr0 = 0.0518
        Cr2 = 0.00035

        # states
        x = states[0]
        y = states[1]
        yaw = states[2]
        vx = states[3]
        vy = states[4]
        omega = states[5]
        # system inputs
        tau = u[0]
        delta = u[1]
        # v_theta = u[2]
        # v_theta = u[2]

        # Slip angles for the front and rear
        alpha_f = -delta - np.arctan((vy + lf * omega) / (vx+0.001))
        alpha_r = np.arctan((-vy + lr * omega) / (vx+0.001))
        # alpha_f = - np.arctan((omega * lf + vy) / (vx + 0.0001)) + delta
        # alpha_r = np.arctan((omega * lr - vy) / (vx + 0.0001))

        Nf = lr / (lr+lf) * m * 9.81
        Nr = lf / (lr+lf) * m * 9.81
        # Lateral Forces
        Ffy = Df * np.sin(Cf * np.arctan(Bf * alpha_f)) * Nf
        Fry = Dr * np.sin(Cr * np.arctan(Br * alpha_r)) * Nr
        # Frx = (Cm1 - Cm2 * vx) * D - Cr0 - Cr2 * vx**2
        Frx = tau / rw

        dx = vx * np.cos(yaw) - vy * np.sin(yaw)
        dy = vx * np.sin(yaw) + vy * np.cos(yaw)
        dyaw = omega
        dvx = 1 / m * (Frx - Ffy * np.sin(delta) + m * vy * omega)
        dvy = 1 / m * (Fry - Ffy * np.cos(delta) - m * vx * omega)
        domega = 1 / Izz * (Ffy * lf * np.cos(delta) - Fry * lr)
        # dtheta = v_theta

        state_dot = [dx, dy, dyaw, dvx, dvy, domega]

        return state_dot

    def controller_cost(self, x, y, yaw):
        """prepares the MPC variables and calculates the cost"""

        # Calculating the contouring cost
        target_index, dx, dy, absolute_error, d = StanleyController.find_target_path_id(self.px, self.py, x, y, yaw,
                                                                                     self.params)
        selected_px = self.px[target_index:target_index + 200]
        selected_py = self.py[target_index:target_index + 200]
        selected_pyaw = self.pyaw[target_index:target_index + 200]
        selected_d = d[target_index:target_index + 200] # lateral error

        # e_c = np.sin(selected_pyaw) * (x - selected_px) - np.cos(selected_pyaw) * (y - selected_py)
        # e_l = -np.cos(selected_pyaw) * (x - selected_px) - np.sin(selected_pyaw) * (y - selected_py)
        # e_y = y - selected_py
        e_y = selected_d
        e_yaw = yaw - selected_pyaw

        qc = 1e-2 * 1 / 0.5 ** 2  # lateral error cost
        qyaw = 1 * 1 / (20 * np.pi / 180) ** 2  # yaw error cost
        # qc = 75
        # qyaw = 500
        # ql = 0.05  # longitudinal error cost
        Q_path = qc
        # Contouring cost
        Jy = 0.
        for e_c_k in e_y:
            Jy += e_c_k.T * Q_path * e_c_k
        Jyaw = 0.
        for e_yaw_k in e_yaw:
            Jyaw += e_yaw_k.T * qyaw * e_yaw_k

        print(f'Lateral error cost is: {Jy}')
        print(f'Yaw error cost is: {Jyaw}')

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
        cost_u = 0.

        # rtau = 1e-6
        # rDelta = 1e-6
        rtau = 0 * 1e-2 * (1 / 1000) ** 2
        rDelta = 0 * 1e-5 * (1 / (10 * np.pi / 180)) ** 2
        # rVs = 1e-6
        R = np.diag([rtau, rDelta])

        # rdtau = 1e-4
        # rdDelta = 5e-3
        rdtau = 1 * 1 * (1 / 1000) ** 2
        rdDelta = 1e6 * (1 / (0.1 * np.pi / 180)) ** 2
        # rdVs = 1e-5
        R_del = np.diag([rdtau, rdDelta])

        # Cost for vx
        R_v = 1 * 1 / 10 ** 2
        cost_v = 0.

        # Lower bound and upper bound on input
        ulb = [-3000, -25. * np.pi / 180]
        uub = [3000., +25. * np.pi / 180]
        for i in range(self.N):
            # states
            s_i = s[nx * i:nx * (i + 1)]
            s_ip1 = s[nx * (i + 1):nx * (i + 2)]  # successor state x_(k+1)
            # inputs
            u_k = q[nu * i:nu * (i + 1)]

            # Decision variable
            zlb += [-np.inf] * nx
            zub += [np.inf] * nx
            zlb += ulb
            zub += uub

            z.append(s_i)
            z.append(u_k)

            if i < self.N - 1:
                u_k1 = q[nu * (i + 1):nu * (i + 2)]  # u_(k+1)
                du_k = u_k1 - u_k
            else:
                du_k = ca.vertcat([0., 0.])

            xt_ip1 = Phi(x0=s_i, p=u_k)['xf']
            # du'*R_du*du + u'*R*u
            if i < self.N - 1:
                cost_u += u_k.T @ R @ u_k + du_k.T @ R_del @ du_k
            else:
                cost_u += u_k.T @ R @ u_k

            # vx * Rv * vx
            cost_v += (s_i[3] - 15) * R_v * (s_i[3] - 15)

            constraints.append(xt_ip1 - s_ip1)
            constraints.append(du_k)

        # s_N
        z.append(s_ip1)
        zlb += [-np.inf] * nx
        zub += [np.inf] * nx
        # total cost
        cost = cost_u + Jy + Jyaw + cost_v
        constraints = ca.vertcat(*constraints)
        variables = ca.vertcat(*z)

        # Create the optimization problem
        g_bnd = np.zeros(self.N * nx)
        nlp = {'f': cost, 'g': constraints, 'x': variables}
        opt = {'print_time': 0, 'ipopt.print_level': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opt)
        return solver, zlb, zub, cost_u, Jy, Jyaw, cost_v

    def solve_mpc(self, current_state):
        """Solve MPC provided the current state, i.e., this
        function is u = h(x), which is the implicit control law of MPC.

        Args:
            current_state (array-like): current state

        Returns:
            tuple: current input and return status pair
        """
        #     # unpacking the real signals that come from the system
        x_r, y_r, yaw_r, vx_r, vy_r, omega_r, ps = current_state

        solver, zlb, zub, cost_u, Jy, Jyaw, cost_v = self.controller_cost(x_r, y_r, yaw_r)
        equality_constraints = np.zeros(self.N * self.nx)
        g_bnd = equality_constraints
        dtau_lb = -300
        dtau_ub = +300
        ddelta_lb = -3 * np.pi/180
        ddelta_ub = +3 * np.pi/180
        g_bnd_lb = []
        g_bnd_ub = []
        for i in range(self.N):
            g_bnd_lb.append([0, 0, 0, 0, 0, 0, dtau_lb, ddelta_lb])
            g_bnd_ub.append([0, 0, 0, 0, 0, 0, dtau_ub, ddelta_ub])

        g_bnd_lb = ca.vertcat(*g_bnd_lb)
        g_bnd_ub = ca.vertcat(*g_bnd_ub)

        # Set the lower and upper bound of the decision variable
        # such that s0 = current_state
        for i in range(self.nx):
            zlb[i] = current_state[i]
            zub[i] = current_state[i]
        sol_out = solver(lbx=zlb, ubx=zub, lbg=g_bnd_lb, ubg=g_bnd_ub)

        u_array = []
        for i in range(self.N):
            initial_slice = self.nx * (i + 1) + self.nu * i
            end_slice = self.nx * (i + 1) + self.nu * (i + 1)
            u = ca.MX.sym(f'u{i}', self.nu, 1)
            u = sol_out['x'][initial_slice:end_slice]

        # cost_u_fun = ca.Function('cost_u', {u}, {cost_u})
        # print(f'Y cost: {Jy}\n')
        # print(f'Yaw cost: {Jyaw}\n')
        # print(f'vx cost: {cost_v}\n')
        # print(f'input cost: {cost_u}\n')

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
