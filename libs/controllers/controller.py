import numpy as np
import casadi as ca
from libs.utils.normalise_angle import normalise_angle
import matplotlib.pyplot as plt
from libs.vehicle_model.vehicle_model import VehicleParameters
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

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
        self.ocp = AcadosOcp()

    def SystemModel(self, param):
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
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        yaw = ca.SX.sym('yaw')
        vx = ca.SX.sym('vx')
        vy = ca.SX.sym('vy')
        omega = ca.SX.sym('omega')
        state = ca.vertcat(x, y, yaw, vx, vy, omega)
        # system inputs
        tau = ca.SX.sym('tau')
        delta = ca.SX.sym('delta')

        # unnamed symoblic variables
        sym_x = ca.vertcat(x, y, yaw, vx, vy, omega)
        sym_xdot = ca.MX.sym('xdot', self.nx, 1)
        sym_u = ca.vertcat(tau, delta)
        x0 = np.zeros(self.nx)


        ## Dynamics
        # Slip angles for the front and rear
        alpha_f = delta - np.arctan((vy + lf * omega) / (vx + 0.001))
        alpha_r = np.arctan((-vy + lr * omega) / (vx + 0.001))

        Nf = lr / (lr + lf) * m * 9.81
        Nr = lf / (lr + lf) * m * 9.81
        # Lateral Forces
        Ffy = 2 * Df * np.sin(Cf * np.arctan(Bf * alpha_f)) * Nf
        Fry = 2 * Dr * np.sin(Cr * np.arctan(Br * alpha_r)) * Nr
        # Frx = (Cm1 - Cm2 * vx) * D - Cr0 - Cr2 * vx**2
        Frx = tau / rw

        dx = vx * np.cos(yaw) - vy * np.sin(yaw)
        dy = vx * np.sin(yaw) + vy * np.cos(yaw)
        dyaw = omega
        dvx = 1 / m * (Frx - Ffy * np.sin(delta) + m * vy * omega)
        dvy = 1 / m * (Fry - Ffy * np.cos(delta) - m * vx * omega)
        domega = 1 / Izz * (Ffy * lf * np.cos(delta) - Fry * lr)
        # dtheta = v_theta

        expr_f_expl = ca.vertcat(dx, dy, dyaw, dvx, dvy, domega)
        expr_f_impl = expr_f_expl - sym_xdot

        ## cost calculations
        qc = 50000 * 1 / 0.5 ** 2  # lateral error cost
        qyaw = 2000 * 1 / (1 * np.pi / 180) ** 2   # yaw error cost
        qv = 10 * 1 / 1 ** 2

        y_r = 0
        yaw_r = 0
        v_r = 15

        cost_expr_y = ca.vertcat(y, yaw, vx)
        y_ref = np.array([y_r, yaw_r, v_r])
        W = np.diag([qc, qyaw, qv])
        cost_expr_y_e = ca.vertcat(y_r, yaw_r, v_r)
        W_e = W
        y_ref_e = y_ref

        # Constraints
        long_acc = dvx - vy * omega
        lat_acc = dvy + vx * omega
        constr_expr_h = ca.vertcat(long_acc, lat_acc)
        bound_h = np.array([9.81, 9.81])
        constr_Jsh = np.eye(2)
        # cost_Z = np.eye(2)
        # cost_z = np.zeros(2, 1)


        model = AcadosModel()
        model.f_impl_expr = expr_f_impl
        model.f_expl_expr = expr_f_expl
        model.x = sym_x
        model.xdot = sym_xdot
        model.u = sym_u

        self.ocp.model = model
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu
        ny_e = nx

        self.ocp.dims.N = self.N
        self.ocp.model.T = self.T
        self.ocp.nx = nx
        self.ocp.nu = nu

        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        self.ocp.W = W
        self.ocp.W_e = W_e
        self.ocp.cost.yref = y_ref
        self.ocp.cost.yref_e = y_ref_e
        self.ocp.model.cost_y_expr = cost_expr_y
        self.ocp.model.cost_y_expr_e = cost_expr_y_e
        self.ocp.constraints.constr_expr_h = constr_expr_h
        self.ocp.constraints.bound_h = bound_h
        self.ocp.constraints.constr_Jsh = constr_Jsh
        self.ocp.solver_options.nlp_solver_type = 'sqp'
        self.ocp.solver_options.qp_solver = 'full_condensing_hpipm'
        self.ocp.solver_options.qp_solver_cond_N = 5
        self.ocp.solver_options.integrator_type = 'ERK'

        return model

    def controller_cost(self, x, y, yaw):
        model = self.SystemModel(self.params)




        return solver, zlb, zub, cost_u, Jy, Jyaw, cost_v

    def solve_mpc(self, current_state, uk_prev_step):
        """Solve MPC provided the current state, i.e., this
        function is u = h(x), which is the implicit control law of MPC.

        Args:
            current_state (array-like): current state

        Returns:
            tuple: current input and return status pair
            :param uk_prev_step: previous input
            :param current_state:
        """
        #     # unpacking the real signals that come from the system
        x_r, y_r, yaw_r, vx_r, vy_r, omega_r, ps = current_state

        solver, zlb, zub, cost_u, Jy, Jyaw, cost_v = self.controller_cost(x_r, y_r, yaw_r)
        equality_constraints = np.zeros(self.N * (self.nx+self.nu))
        g_bnd = equality_constraints

        # Set the lower and upper bound of the decision variable
        # such that s0 = current_state
        current_xtilda_state = np.concatenate([np.array(current_state[0:6]), uk_prev_step[:]])
        for i in range(self.nx+self.nu):
            zlb[i] = current_xtilda_state[i]
            zub[i] = current_xtilda_state[i]
        sol_out = solver(lbx=zlb, ubx=zub, lbg=g_bnd, ubg=g_bnd)

        # u_array = []
        # for i in range(self.N):
        #     initial_slice = self.nx * (i + 1) + self.nu * i
        #     end_slice = self.nx * (i + 1) + self.nu * (i + 1)
        #     u = ca.MX.sym(f'u{i}', self.nu, 1)
        #     u = sol_out['x'][initial_slice:end_slice]

        # cost_u_fun = ca.Function('cost_u', {u}, {cost_u})
        # print(f'Y cost: {Jy}\n')
        # print(f'Yaw cost: {Jyaw}\n')
        # print(f'vx cost: {cost_v}\n')
        # print(f'input cost: {cost_u}\n')

        return np.array(sol_out['x'][2*self.nx+2*self.nu:2*self.nx + 3*self.nu]), solver.stats()['return_status']


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
