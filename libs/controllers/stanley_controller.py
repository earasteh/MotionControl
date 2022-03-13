import numpy as np
import casadi as ca
from libs.utils.normalise_angle import normalise_angle
import matplotlib.pyplot as plt

# Path interpolation parameters
INTERP_MAX_POINTS_PLOT = 2  # number of points used for displaying
# selected path
INTERP_DISTANCE_RES = 0.1  # distance between interpolated points


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

    def find_target_path_id(self, x, y, yaw):
        # Calculate position of the front axle
        fx = x + self.L * np.cos(yaw)
        fy = y + self.L * np.sin(yaw)

        dx = fx - self.px  # Find the x-axis of the front axle relative to the path
        dy = fy - self.py  # Find the y-axis of the front axle relative to the path

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
        target_index, dx, dy, absolute_error = self.find_target_path_id(x, y, yaw)
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


class MPCC(StanleyController):
    def __init__(self, N, T, p):
        self.N = N # dt = T/N
        self.T = T # Total time
        self.p = p # system parameters
        self.controller_prep()

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

    def controller_prep(self):
    #     # unpacking the real signals that come from the system
    #     x_r, y_r, yaw_r, vx_r, vy_r, yaw_dot_r = x_real

        dt = self.T / self.N
        # CasADi works with symbolics
        t = ca.SX.sym("t", 1, 1)
        x = ca.SX.sym("x", 6, 1)
        u = ca.SX.sym("u", 3, 1)
        ode = ca.vertcat(*self.SystemModel(x, u, self.p))
        f = {'x': x, 't': t, 'p': u, 'ode': ode}
        Phi = ca.integrator("Phi", "cvodes", f, {'tf': dt})



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
