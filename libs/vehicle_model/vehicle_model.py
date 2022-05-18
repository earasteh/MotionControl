#!/usr/bin/env python
from numpy import cos, sin, tan, clip, abs, sqrt, arctan, pi, array, linspace
import numpy as np
from libs.utils.normalise_angle import normalise_angle
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Empty lists for global coordinates Vx, Vy, ax, and ay
Vx_planar = []
Vy_planar = []
Ax_planar = []
Ay_planar = []
t_planar = []

Vx_mpc = []
Vy_mpc = []
Ax_mpc = []
Ay_mpc = []
t_mpc = []


class VehicleParameters:
    def __init__(self, mf=987.89, mr=869.93, mus=50, L=2.906,
                 ab_ratio=0.85, T=1.536, hg=0.55419, Jw=1,
                 kf=26290, kr=25830,
                 Efront=0.0376, Erear=0, LeverArm=0.13256,
                 BFL=20.6357, CFL=1.5047, DFL=1.1233):
        self.rr = 0.329  # rolling radius
        self.mus = mus  # unsprung mass (one wheel)
        self.mf = mf  # front axle mass
        self.mr = mr  # rear axle mass
        self.m = mf + mr  # mass
        self.L = L  # length of the car
        self.ab_ratio = ab_ratio
        self.b = self.L / (1 + self.ab_ratio)
        self.a = self.L - self.b
        self.Izz = 0.5 * self.m * self.a * self.b  # moment of ineria z-axis
        self.Jw = Jw  # wheel's inertia
        self.hg = hg  # Height of mass centre above ground (m)
        self.T = T  # track width
        self.kf = kf  # front suspension stiffness
        self.kr = kr  # rear suspension stiffness
        self.rw = self.rr - (self.mf / 2 + self.mus) / self.kf

        ## Tire parameters for Magic formula
        self.BFL = BFL
        self.CFL = CFL
        self.DFL = DFL
        self.BFR = self.BFL
        self.CFR = self.CFL
        self.DFR = self.DFL

        self.BRL = self.BFL
        self.CRL = self.CFL
        self.DRL = self.DFL

        self.BRR = self.BFL
        self.CRR = self.CFL
        self.DRR = self.DFL

        # self.Efront = Efront  # Trail/Front Wheel (m)
        # self.Erear = Erear  # Trail/Rear Wheel (m)
        # self.E = [self.Efront, self.Efront, self.Erear, self.Erear]
        # self.LeverArm = LeverArm
        self.wL = self.T / 2
        self.wR = self.T / 2


param = VehicleParameters()  # parameters


class VehicleModel:

    def __init__(self, wheelbase=1.0, max_steer=0.7, dt=0.05):
        """
        2D Kinematic Bicycle Model

        At initialisation
        :param wheelbase:       (float) vehicle's wheelbase [m]
        :param max_steer:       (float) vehicle's steering limits [rad]
        :param dt:              (float) discrete time period [s]
    
        At every time step  
        :param x:               (float) vehicle's x-coordinate [m]
        :param y:               (float) vehicle's y-coordinate [m]
        :param yaw:             (float) vehicle's heading [rad]
        :param velocity:        (float) vehicle's velocity in the x-axis [m/s]
        :param throttle:        (float) vehicle's accleration [m/s^2]
        :param delta:           (float) vehicle's steering angle [rad]
    
        :return x:              (float) vehicle's x-coordinate [m]
        :return y:              (float) vehicle's y-coordinate [m]
        :return yaw:            (float) vehicle's heading [rad]
        :return velocity:       (float) vehicle's velocity in the x-axis [m/s]
        :return delta:          (float) vehicle's steering angle [rad]
        :return omega:          (float) vehicle's angular velocity [rad/s]
        """
        self.dt = dt
        self.wheelbase = wheelbase
        self.max_steer = max_steer

    def planar_model(self, state, tire_torques, mu_max, delta, p, ax_prev, ay_prev):
        """:This is the function for 7 ِِDoF model with nonlinear tires (pacejka magic formula)
        """
        # Unpacking the state-space and inputs
        U, V, wz, wFL, wFR, wRL, wRR, yaw, x, y = state
        deltaFL, deltaFR, deltaRL, deltaRR = delta
        TFL, TFR, TRL, TRR = tire_torques
        mumaxFL, mumaxFR, mumaxRL, mumaxRR = mu_max

        # Parameters
        g = 9.81

        p.DFL = mumaxFL
        p.DFR = mumaxFR
        p.DRL = mumaxRL
        p.DRR = mumaxRR

        # Messing with the tire parameters (under/oversteer)
        # p.CRR = 0.8 * p.CFR
        # p.CRL = 0.8 * p.CFL
        #
        # p.BRR = 0.8 * p.BFR
        # p.BRL = 0.8 * p.BFL

        ## Normal forces (static forces)
        fFLz0 = p.b / (p.a + p.b) * p.m * g / 2
        fFRz0 = p.b / (p.a + p.b) * p.m * g / 2
        fRLz0 = p.a / (p.a + p.b) * p.m * g / 2
        fRRz0 = p.a / (p.a + p.b) * p.m * g / 2

        DfzxL = p.m * p.hg * p.wR / ((p.a + p.b) * (p.wL + p.wR))
        DfzxR = p.m * p.hg * p.wL / ((p.a + p.b) * (p.wL + p.wR))
        DfzyF = p.m * p.hg * p.b / ((p.a + p.b) * (p.wL + p.wR))
        DfzyR = p.m * p.hg * p.a / ((p.a + p.b) * (p.wL + p.wR))

        fFLz = fFLz0  # - DfzxL * ax_prev - DfzyF * ay_prev
        fFRz = fFRz0  # - DfzxR * ax_prev + DfzyF * ay_prev
        fRLz = fRLz0  # + DfzxL * ax_prev - DfzyR * ay_prev
        fRRz = fRRz0  # + DfzxR * ax_prev + DfzyR * ay_prev

        ## Compute tire slip Wheel velocities
        vFLxc = U - p.T * wz / 2
        vFLyc = V + p.a * wz

        vFRxc = U + p.T * wz / 2
        vFRyc = V + p.a * wz

        vRLxc = U - p.T * wz / 2
        vRLyc = V - p.b * wz

        vRRxc = U + p.T * wz / 2
        vRRyc = V - p.b * wz

        # Rotate to obtain velocities in the tire frame
        vFLx = vFLxc * cos(deltaFL) + vFLyc * sin(deltaFL)
        vFLy = -vFLxc * sin(deltaFL) + vFLyc * cos(deltaFL)
        vFRx = vFRxc * cos(deltaFR) + vFRyc * sin(deltaFR)
        vFRy = -vFRxc * sin(deltaFR) + vFRyc * cos(deltaFR)
        vRLx = vRLxc * cos(deltaRL) + vRLyc * sin(deltaRL)
        vRLy = -vRLxc * sin(deltaRL) + vRLyc * cos(deltaRL)
        vRRx = vRRxc * cos(deltaRR) + vRRyc * sin(deltaRR)
        vRRy = -vRRxc * sin(deltaRR) + vRRyc * cos(deltaRR)

        ## Longitudinal slip
        sFLx = p.rw * wFL / vFLx - 1
        sFRx = p.rw * wFR / vFRx - 1
        sRLx = p.rw * wRL / vRLx - 1
        sRRx = p.rw * wRR / vRRx - 1

        ## Lateral slip
        sFLy = -vFLy / abs(vFLx)
        sFRy = -vFRy / abs(vFRx)
        sRLy = -vRLy / abs(vRLx)
        sRRy = -vRRy / abs(vRRx)

        ## Combined slip
        sFL = sqrt(sFLx ** 2 + sFLy ** 2)
        sFR = sqrt(sFRx ** 2 + sFRy ** 2)
        sRL = sqrt(sRLx ** 2 + sRLy ** 2)
        sRR = sqrt(sRRx ** 2 + sRRy ** 2)

        # Compute tire forces
        ## Combined friction coefficient
        muFL = p.DFL * sin(p.CFL * arctan(p.BFL * sFL))
        muFR = p.DFR * sin(p.CFR * arctan(p.BFR * sFR))
        muRL = p.DRL * sin(p.CRL * arctan(p.BRL * sRL))
        muRR = p.DRR * sin(p.CRR * arctan(p.BRR * sRR))

        ## Longitudinal friction coefficient
        if sFL != 0:
            muFLx = sFLx * muFL / sFL
        else:
            muFLx = p.DFL * sin(p.CFL * arctan(p.BFL * sFLx))

        if sFR != 0:
            muFRx = sFRx * muFR / sFR
        else:
            muFRx = p.DFR * sin(p.CFR * arctan(p.BFR * sFRx))

        if sRL != 0:
            muRLx = sRLx * muRL / sRL
        else:
            muRLx = p.DRL * sin(p.CRL * arctan(p.BRL * sRLx))

        if sRR != 0:
            muRRx = sRRx * muRR / sRR
        else:
            muRRx = p.DRR * sin(p.CRR * arctan(p.BRR * sRRx))

        ## Lateral Friction coefficient
        if sFL != 0:
            muFLy = sFLy * muFL / sFL
        else:
            muFLy = p.DFL * sin(p.CFL * arctan(p.BFL * sFLy))

        if sFR != 0:
            muFRy = sFRy * muFR / sFR
        else:
            muFRy = p.DFR * sin(p.CFR * arctan(p.BFR * sFRy))

        if sRL != 0:
            muRLy = sRLy * muRL / sRL
        else:
            muRLy = p.DRL * sin(p.CRL * arctan(p.BRL * sRLy))

        if sRR != 0:
            muRRy = sRRy * muRR / sRR
        else:
            muRRy = p.DRR * sin(p.CRR * arctan(p.BRR * sRRy))

        ## Compute longitudinal force
        fFLxt = muFLx * fFLz
        fFRxt = muFRx * fFRz
        fRLxt = muRLx * fRLz
        fRRxt = muRRx * fRRz

        ## Compute lateral forces
        fFLyt = muFLy * fFLz
        fFRyt = muFRy * fFRz
        fRLyt = muRLy * fRLz
        fRRyt = muRRy * fRRz

        ## Rotate to obtain forces in the chassis frame
        fFLx = fFLxt * cos(deltaFL) - fFLyt * sin(deltaFL)
        fFLy = fFLxt * sin(deltaFL) + fFLyt * cos(deltaFL)

        fFRx = fFRxt * cos(deltaFR) - fFRyt * sin(deltaFR)
        fFRy = fFRxt * sin(deltaFR) + fFRyt * cos(deltaFR)

        fRLx = fRLxt * cos(deltaRL) - fRLyt * sin(deltaRL)
        fRLy = fRLxt * sin(deltaRL) + fRLyt * cos(deltaRL)

        fRRx = fRRxt * cos(deltaRR) - fRRyt * sin(deltaRR)
        fRRy = fRRxt * sin(deltaRR) + fRRyt * cos(deltaRR)

        # Compute the time derivatives
        U_dot = 1 / p.m * (fFLx + fFRx + fRLx + fRRx) + V * wz
        V_dot = 1 / p.m * (fFLy + fFRy + fRLy + fRRy) - U * wz
        wz_dot = 1 / p.Izz * (p.a * (fFLy + fFRy) - p.b * (fRLy + fRRy) + p.T / 2 * (fFRx - fFLx + fRRx - fRLx))
        wFL_dot = (TFL - p.rw * fFLxt) / p.Jw
        wFR_dot = (TFR - p.rw * fFRxt) / p.Jw
        wRL_dot = (TRL - p.rw * fRLx) / p.Jw
        wRR_dot = (TRR - p.rw * fRRx) / p.Jw
        yaw_dot = wz
        x_dot = U * cos(yaw) - V * sin(yaw)
        y_dot = U * sin(yaw) + V * cos(yaw)

        state_dot = np.array([U_dot, V_dot, wz_dot, wFL_dot, wFR_dot, wRL_dot, wRR_dot, yaw_dot, x_dot, y_dot])

        # Velocities at the wheel contact patch in the global inertial frame
        # vFLxg = vFLxc * cos(yaw) - vFLyc * sin(yaw)
        # vFRxg = vFRxc * cos(yaw) - vFRyc * sin(yaw)
        # vRLxg = vRLxc * cos(yaw) - vRLyc * sin(yaw)
        # vRRxg = vRRxc * cos(yaw) - vRRyc * sin(yaw)
        # vFLyg = vFLxc * sin(yaw) + vFLyc * cos(yaw)
        # vFRyg = vFRxc * sin(yaw) + vFRyc * cos(yaw)
        # vRLyg = vRLxc * sin(yaw) + vRLyc * cos(yaw)
        # vRRyg = vRRxc * sin(yaw) + vRRyc * cos(yaw)

        # Position of the wheel contact patch in the global inertial frame
        # xFL = x + p.a * cos(yaw) - p.T / 2 * sin(yaw)
        # yFL = y + p.a * sin(yaw) + p.T / 2 * cos(yaw)
        # xFR = x + p.a * cos(yaw) + p.T / 2 * sin(yaw)
        # yFR = y + p.a * sin(yaw) - p.T / 2 * cos(yaw)
        # xRL = x - p.b * cos(yaw) - p.T / 2 * sin(yaw)
        # yRL = y - p.b * sin(yaw) + p.T / 2 * cos(yaw)
        # xRR = x - p.b * cos(yaw) + p.T / 2 * sin(yaw)
        # yRR = y - p.b * sin(yaw) - p.T / 2 * cos(yaw)

        # Chassis velocity, and acceleration in the global inertial frame
        vx = U * cos(yaw) - V * sin(yaw)
        vy = V * sin(yaw) + U * cos(yaw)

        axc = U_dot - V * wz
        ayc = V_dot + U * wz

        # global frame ax and ay
        ax = axc * cos(yaw) - ayc * sin(yaw)
        ay = axc * sin(yaw) + ayc * cos(yaw)

        yaw = normalise_angle(yaw)
        # state_update = [U, V, wz, wFL, wFR, wRL, wRR, yaw, x, y]
        outputs = np.array([fFLx, fFRx, fRLx, fRRx,
                            fFLy, fFRy, fRLy, fRRy,
                            fFLz, fFRz, fRLz, fRRz,
                            sFL, sFR, sRL, sRR, fFLxt, fFLyt])

        return [state_dot, vx, vy, ax, ay, outputs, axc, ayc]

    def planar_model_RK4(self, state, tire_torques, mu_max, delta, p, ax_prev, ay_prev):
        h = self.dt
        K1, _, _, _, _, outputs1, axc1, ayc1 = self.planar_model(state, tire_torques, mu_max, delta, p,
                                                                 ax_prev, ay_prev)
        K2, _, _, _, _, outputs2, axc2, ayc2 = self.planar_model(np.array(state) + h / 2 * K1, tire_torques, mu_max,
                                                                 delta, p, ax_prev, ay_prev)
        K3, _, _, _, _, outputs3, axc3, ayc3 = self.planar_model(np.array(state) + h / 2 * K2, tire_torques, mu_max,
                                                                 delta, p, ax_prev, ay_prev)
        K4, _, _, _, _, outputs4, axc4, ayc4 = self.planar_model(np.array(state) + h * K3, tire_torques, mu_max,
                                                                 delta, p, ax_prev, ay_prev)

        state_update = state + 1 / 6 * h * (K1 + 2 * K2 + 2 * K3 + K4)
        U, V, wz, wFL, wFR, wRL, wRR, yaw, x, y = state_update
        state_dot = (K1 + 2 * K2 + 2 * K3 + K4) / 6
        outputs = (outputs1 + 2 * outputs2 + 2 * outputs3 + outputs4) / 6
        axc = (axc1 + 2 * axc2 + 2 * axc3 + axc4) / 6
        ayc = (ayc1 + 2 * ayc2 + 2 * ayc3 + ayc4) / 6

        return [state_update, x, y, yaw, U, state_dot, outputs, axc, ayc]

    def SystemModel(self, t, states, u, param, ep1, ep2):
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

        # Slip angles for the front and rear
        alpha_f = -delta - np.arctan((vy + lf * omega) / (vx))
        alpha_r = np.arctan((-vy + lr * omega) / (vx))
        # alpha_f = - np.arctan((omega * lf + vy) / (vx + 0.0001)) + delta
        # alpha_r = np.arctan((omega * lr - vy) / (vx + 0.0001))

        Nf = lr / (lr + lf) * m * 9.81
        Nr = lf / (lr + lf) * m * 9.81
        # Lateral Forces
        Ffy = 2 * Df * np.sin(ep1 * Cf * np.arctan(ep2 * Bf * alpha_f)) * Nf
        Fry = 2 * Dr * np.sin(ep1 * Cr * np.arctan(ep2 * Br * alpha_r)) * Nr
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


def planar_integrate(t, state, tire_torques, mu_max, delta, p):
    if t == 0:
        axc_prev = 0
        ayc_prev = 0

    veh = VehicleModel()
    [state_dot, vx, vy, ax, ay, outputs, axc_next, ayc_next] = veh.planar_model(state, tire_torques, mu_max, delta, p,
                                                                                0, 0)
    axc_prev = axc_next
    ayc_prev = ayc_next

    Vx_planar.append(vx)
    Vy_planar.append(vy)
    Ax_planar.append(ax)
    Ay_planar.append(ay)
    t_planar.append(t)
    return state_dot


def main():
    # print("This script is not meant to be executable, and should be used as a library.")
    p1 = VehicleParameters()
    U_init = 8.33
    state0_planar = [U_init, 0, 0,  # U, V, wz
                     U_init / p1.rw, U_init / p1.rw, U_init / p1.rw, U_init / p1.rw,  # wFL, wFR, wRL, wRR
                     0, 0, 0]  # yaw, x, y
    print("Initial condition:", state0_planar)

    tire_torques = [0, 0, 0, 0]
    delta = np.pi / 180 * array([2, 2, 0, 0])
    mu_max = [1, 1, 1, 1]  # road surface maximum friction

    # planar_integrate(state0, tire_torques, mu_max, delta, p1)
    t_array = linspace(0, 5, 100)
    sol_planar = solve_ivp(planar_integrate, [0, 5], state0_planar, args=(tire_torques, mu_max, delta, p1),
                           method='RK45',
                           dense_output=True,
                           t_eval=t_array)

    t_planar = sol_planar.t
    U_planar, V_planar, wz_planar, wFL_planar, wFR_planar, wRL_planar, wRR_planar, yaw_planar, x_planar, y_planar = sol_planar.y

    # SystemModel(self, t, states, u, param, xs=None, us=None)
    state0_mpc = [0, 0, 0, U_init, 0, 0]
    veh = VehicleModel()
    ## old mpc model

    ## Optimize
    def ompc_diff(ep, veh, t, x_p, y_p, param):
        ep1, ep2 = ep
        sol_oldmpc = solve_ivp(veh.SystemModel, [0, 5], state0_mpc, args=([tire_torques[0], delta[0]], param, ep1, ep2),
                               method='RK45',
                               dense_output=True,
                               t_eval=t)
        t_oldmpc = sol_oldmpc.t
        x_ompc, y_ompc, yaw_ompc, U_ompc, V_ompc, omega_ompc = sol_oldmpc.y

        d_ompc = np.sqrt((x_ompc[-1] - x_p[-1]) ** 2 + (y_ompc[-1] - y_p[-1]) ** 2)

        return d_ompc

    min_ep = minimize(ompc_diff, x0=np.array([1.5, 1]), args=(veh, t_array, x_planar, y_planar, p1), bounds=((0.1, 2), (0.1, 2)))
    print(min_ep.x)
    print(ompc_diff([1.36, 2.0], veh, t_array, x_planar, y_planar, p1))

    ## compute with  optimized values
    sol_oldmpc = solve_ivp(veh.SystemModel, [0, 5], state0_mpc, args=([tire_torques[0], delta[0]], param, 1.365, 2.0),
                           method='RK45',
                           dense_output=True,
                           t_eval=t_array)
    t_oldmpc = sol_oldmpc.t
    x_ompc, y_ompc, yaw_ompc, U_ompc, V_ompc, omega_ompc = sol_oldmpc.y
    d_ompc = np.sqrt((x_ompc[-1] - x_planar[-1]) ** 2 + (y_ompc[-1] - y_planar[-1]) ** 2)

    ## MPC Model
    from libs.controllers.controller import MPC
    from acados_template import AcadosSim, AcadosSimSolver

    px = np.linspace(0, 100, 1000)
    py = np.zeros((1, 1000))
    pyaw = np.zeros((1, 1000))
    init = np.array([0, 0, 0, U_init, 0, 0])
    N = 500
    Tf = 0.01

    test_mpc = MPC(N, Tf, param=p1, px=px, py=py, pyaw=pyaw, veh_initial_conditions=state0_mpc)

    sim = AcadosSim()
    model, constraint = test_mpc.bicycle_model(init, p1)
    sim.model = model
    sim.model.dyn_ext_fun_type = 'casadi'

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
    u0 = np.array([0.0, 2.0 * np.pi/180])
    acados_integrator.set("u", u0)

    simX[0, :] = x0

    for i in range(N):
        # set initial state
        acados_integrator.set("x", simX[i, :])
        # initialize IRK
        if sim.solver_options.integrator_type == 'IRK':
            acados_integrator.set("xdot", np.zeros((nx,)))

        # solve
        status = acados_integrator.solve()
        # get solution
        simX[i + 1, :] = acados_integrator.get("x")

    if status != 0:
        raise Exception('acados returned status {}. Exiting.'.format(status))


    t = np.linspace(0, N*Tf, N + 1)

    plt.figure()
    plt.title('x-y position')
    plt.plot(x_planar, y_planar)
    plt.plot(simX[:, 0], simX[:, 1])
    plt.plot(x_ompc, y_ompc)
    plt.legend(['Planar model', 'MPC model', 'Old MPC'])
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    d_mpc = np.sqrt((simX[-1, 0] - x_planar[-1]) ** 2 + (simX[-1, 1] - y_planar[-1]) ** 2)
    print(f'diff: {d_ompc}')
    print(f'nmpc diff: {d_mpc}')

    plt.figure()
    plt.title('yaw rate')
    plt.xlabel('Time (Sec)')
    plt.ylabel('Yaw rate (rad/sec)')
    plt.plot(t_planar, wz_planar)
    plt.plot(t, simX[:, 5])
    plt.plot(t_oldmpc, omega_ompc)
    plt.legend(['Planar model', 'MPC model', 'Old MPC'])
    plt.show()


if __name__ == "__main__":
    main()
