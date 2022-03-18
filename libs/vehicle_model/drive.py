import os
import numpy as np
from libs.vehicle_model.vehicle_model import VehicleModel
from libs.vehicle_model.vehicle_model import VehicleParameters
from libs.controllers.controller import StanleyController, MPCC
from libs.controllers.controller import LongitudinalController
from libs.utils.env import world

###
# Frame rate = 0.1
# Vehicle simulation time = 1e-3
# Controller time = 1e-2
# ###

Veh_SIM_NUM = 10  # Number of times vehicle simulation (Simulation_resolution  = sim.dt/Veh_SIM_NUM)
Control_SIM_NUM = Veh_SIM_NUM / 10

param = VehicleParameters()


class Car:

    def __init__(self, init_x, init_y, init_yaw, px, py, pyaw, ps, dt):
        # Variable to log all the data
        self.DataLog = np.zeros((Veh_SIM_NUM * 4000, 45))
        # Model parameters
        init_vel = 10.0
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.prev_vel = self.v = init_vel
        self.target_vel = init_vel
        self.total_vel_error = 0
        self.delta = 0.0
        self.omega = 0.0
        self.wheelbase = 2.906
        self.max_steer = np.deg2rad(30)
        self.dt = dt
        self.c_r = 0.01
        self.c_a = 2.0
        self.ax_prev = 0
        self.ay_prev = 0

        # self.state = [init_vel, 0, 0, init_yaw, init_x, init_y] (these were for the bicycle model states)
        self.state = [init_vel, 0, 0, init_vel / param.rw, init_vel / param.rw, init_vel / param.rw,
                      init_vel / param.rw, init_yaw,
                      init_x, init_y]
        self.state_dot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # path data
        self.px = px
        self.py = py
        self.pyaw = pyaw
        self.ps = ps  # path parameter (x(s), y(s), yaw(s))
        # Lateral Tracker parameters
        self.k = 1
        self.ksoft = 0.01
        self.kyaw = 0
        self.ksteer = 0
        self.crosstrack_error = None
        self.target_id = None
        self.x_del = [0]
        self._prev_paths = 0
        self._prev_best_index = 0
        self._prev_best_path = None

        # Longitudinal Tracker parameters
        self.k_v = 1000
        self.k_i = 100
        self.k_d = 0
        self.torque_vec = [0, 0, 0, 0]

        # Description parameters
        self.overall_length = 4.97
        self.overall_width = 1.964
        self.tyre_diameter = 0.4826
        self.tyre_width = 0.2032
        self.axle_track = 1.662
        self.rear_overhang = (self.overall_length - self.wheelbase) / 2
        self.colour = 'black'
        self.lateral_tracker = StanleyController(self.k, self.ksoft, self.kyaw, self.ksteer, self.max_steer,
                                                 self.wheelbase, param, waypoints=[px, py, pyaw])
        self.kbm = VehicleModel(self.wheelbase, self.max_steer, self.dt)
        self.long_tracker = LongitudinalController(self.k_v, self.k_i, self.k_d)
        T = 0.05
        dt = 0.01
        N = int(T/dt)
        print(f'N is {N}')
        self.MPC = MPCC(N, T, param, self.px, self.py, self.pyaw)
        self.uk_prev_step = np.array([10, -0.5 * np.pi / 180])

    def drive(self, frame):
        # Motion Planner:
        for i in range(Veh_SIM_NUM):
            ## Motion Controllers:
            if i % 10 == 0:
                # self.delta, self.target_id, self.crosstrack_error = \
                #     self.lateral_tracker.stanley_control(self.x, self.y, self.yaw, self.v)
                # self.total_vel_error, self.torque_vec = \
                #     self.long_tracker.long_control(self.target_vel, self.v, self.prev_vel,
                #                                    self.total_vel_error, self.dt)
                # self.prev_vel = self.v
                # self.MPC.controller()
                out, status = self.MPC.solve_mpc([self.x, self.y, self.yaw, self.v, self.state[1], self.state_dot[7]],
                                                 self.uk_prev_step)
                self.uk_prev_step = out.reshape((2,))
                tau, delta = out.tolist()
                self.delta = delta[0]
                self.torque_vec = [tau[0]] * 4
                print(f'Solver status: {status} \n')
                print(f'delta: {self.delta * 180 / np.pi} \n')
                print(f'tau: {self.torque_vec[0]} \n')
                self.lateral_tracker.update_waypoints()

                # Filter the delta output
                # self.x_del.append((1 - 1e-5 / (2 * 0.001)) * self.x_del[-1] + 1e-5 / (2 * 0.001) * self.delta)
                # self.delta = self.x_del[-1]

            ## Vehicle model
            self.state, self.x, self.y, self.yaw, self.v, self.state_dot, outputs, self.ax_prev, self.ay_prev = \
                self.kbm.planar_model_RK4(self.state, self.torque_vec, [1.0, 1.0, 1.0, 1.0],
                                          [self.delta, self.delta, 0, 0], param, self.ax_prev, self.ay_prev)

            self.DataLog[frame * Veh_SIM_NUM + i, 0] = (frame * Veh_SIM_NUM + i) * self.kbm.dt
            self.DataLog[frame * Veh_SIM_NUM + i, 1:11] = self.state
            self.DataLog[frame * Veh_SIM_NUM + i, 11:21] = self.state_dot
            self.DataLog[frame * Veh_SIM_NUM + i, 21] = self.delta
            self.DataLog[frame * Veh_SIM_NUM + i, 22:26] = self.torque_vec
            self.DataLog[frame * Veh_SIM_NUM + i, 26:44] = outputs
            self.DataLog[frame * Veh_SIM_NUM + i, 44] = self.crosstrack_error

        os.system('cls' if os.name == 'nt' else 'clear')
