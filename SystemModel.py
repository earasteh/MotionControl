# Class for all the vehicle parameters
import numpy as np

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

        self.Efront = Efront  # Trail/Front Wheel (m)
        self.Erear = Erear  # Trail/Rear Wheel (m)
        self.E = [self.Efront, self.Efront, self.Erear, self.Erear]
        self.LeverArm = LeverArm
        self.wL = self.T / 2
        self.wR = self.T / 2


param = VehicleParameters()  # all the current parameters


# Vehicle model class
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
        inputs : tire torques (each tire), delta (each tire)
        parameters: mu_max (each tire), p (all the parameter settings defined in the parameter class)
        states: U, V, wz, wFR, wRL, wRR, yaw, x, y
        ax_prev, ay_prev: previous step of forward and lateral acceleration used to calculate the normal forces
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

        fFLz = fFLz0 - DfzxL * ax_prev - DfzyF * ay_prev
        fFRz = fFRz0 - DfzxR * ax_prev + DfzyF * ay_prev
        fRLz = fRLz0 + DfzxL * ax_prev - DfzyR * ay_prev
        fRRz = fRRz0 + DfzxR * ax_prev + DfzyR * ay_prev

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
        vFLx = vFLxc * np.cos(deltaFL) + vFLyc * np.sin(deltaFL)
        vFLy = -vFLxc * np.sin(deltaFL) + vFLyc * np.cos(deltaFL)
        vFRx = vFRxc * np.cos(deltaFR) + vFRyc * np.sin(deltaFR)
        vFRy = -vFRxc * np.sin(deltaFR) + vFRyc * np.cos(deltaFR)
        vRLx = vRLxc * np.cos(deltaRL) + vRLyc * np.sin(deltaRL)
        vRLy = -vRLxc * np.sin(deltaRL) + vRLyc * np.cos(deltaRL)
        vRRx = vRRxc * np.cos(deltaRR) + vRRyc * np.sin(deltaRR)
        vRRy = -vRRxc * np.sin(deltaRR) + vRRyc * np.cos(deltaRR)

        ## Longitudinal slip
        sFLx = p.rw * wFL / vFLx - 1
        sFRx = p.rw * wFR / vFRx - 1
        sRLx = p.rw * wRL / vRLx - 1
        sRRx = p.rw * wRR / vRRx - 1

        ## Lateral slip
        sFLy = -vFLy / np.abs(vFLx)
        sFRy = -vFRy / np.abs(vFRx)
        sRLy = -vRLy / np.abs(vRLx)
        sRRy = -vRRy / np.abs(vRRx)

        ## Combined slip
        sFL = np.sqrt(sFLx ** 2 + sFLy ** 2)
        sFR = np.sqrt(sFRx ** 2 + sFRy ** 2)
        sRL = np.sqrt(sRLx ** 2 + sRLy ** 2)
        sRR = np.sqrt(sRRx ** 2 + sRRy ** 2)

        # Compute tire forces
        ## Combined friction coefficient
        muFL = p.DFL * np.sin(p.CFL * np.arctan(p.BFL * sFL))
        muFR = p.DFR * np.sin(p.CFR * np.arctan(p.BFR * sFR))
        muRL = p.DRL * np.sin(p.CRL * np.arctan(p.BRL * sRL))
        muRR = p.DRR * np.sin(p.CRR * np.arctan(p.BRR * sRR))

        ## Longitudinal friction coefficient
        if sFL != 0:
            muFLx = sFLx * muFL / sFL
        else:
            muFLx = p.DFL * np.sin(p.CFL * np.arctan(p.BFL * sFLx))

        if sFR != 0:
            muFRx = sFRx * muFR / sFR
        else:
            muFRx = p.DFR * np.sin(p.CFR * np.arctan(p.BFR * sFRx))

        if sRL != 0:
            muRLx = sRLx * muRL / sRL
        else:
            muRLx = p.DRL * np.sin(p.CRL * np.arctan(p.BRL * sRLx))

        if sRR != 0:
            muRRx = sRRx * muRR / sRR
        else:
            muRRx = p.DRR * np.sin(p.CRR * np.arctan(p.BRR * sRRx))

        ## Lateral Friction coefficient
        if sFL != 0:
            muFLy = sFLy * muFL / sFL
        else:
            muFLy = p.DFL * np.sin(p.CFL * np.arctan(p.BFL * sFLy))

        if sFR != 0:
            muFRy = sFRy * muFR / sFR
        else:
            muFRy = p.DFR * np.sin(p.CFR * np.arctan(p.BFR * sFRy))

        if sRL != 0:
            muRLy = sRLy * muRL / sRL
        else:
            muRLy = p.DRL * np.sin(p.CRL * np.arctan(p.BRL * sRLy))

        if sRR != 0:
            muRRy = sRRy * muRR / sRR
        else:
            muRRy = p.DRR * np.sin(p.CRR * np.arctan(p.BRR * sRRy))

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
        fFLx = fFLxt * np.cos(deltaFL) - fFLyt * np.sin(deltaFL)
        fFLy = fFLxt * np.sin(deltaFL) + fFLyt * np.cos(deltaFL)

        fFRx = fFRxt * np.cos(deltaFR) - fFRyt * np.sin(deltaFR)
        fFRy = fFRxt * np.sin(deltaFR) + fFRyt * np.cos(deltaFR)

        fRLx = fRLxt * np.cos(deltaRL) - fRLyt * np.sin(deltaRL)
        fRLy = fRLxt * np.sin(deltaRL) + fRLyt * np.cos(deltaRL)

        fRRx = fRRxt * np.cos(deltaRR) - fRRyt * np.sin(deltaRR)
        fRRy = fRRxt * np.sin(deltaRR) + fRRyt * np.cos(deltaRR)

        # Compute the time derivatives
        U_dot = 1 / p.m * (fFLx + fFRx + fRLx + fRRx) + V * wz
        V_dot = 1 / p.m * (fFLy + fFRy + fRLy + fRRy) - U * wz
        wz_dot = 1 / p.Izz * (p.a * (fFLy + fFRy) - p.b * (fRLy + fRRy) + p.T / 2 * (fFRx - fFLx + fRRx - fRLx))
        wFL_dot = (TFL - p.rw * fFLxt) / p.Jw
        wFR_dot = (TFR - p.rw * fFRxt) / p.Jw
        wRL_dot = (TRL - p.rw * fRLx) / p.Jw
        wRR_dot = (TRR - p.rw * fRRx) / p.Jw
        yaw_dot = wz
        x_dot = U * np.cos(yaw) - V * np.sin(yaw)
        y_dot = U * np.sin(yaw) + V * np.cos(yaw)

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
        vx = U * np.cos(yaw) - V * np.sin(yaw)
        vy = V * np.sin(yaw) + U * np.cos(yaw)

        axc = U_dot - V * wz
        ayc = V_dot + U * wz
        ax = axc * np.cos(yaw) - ayc * np.sin(yaw)
        ay = axc * np.sin(yaw) + ayc * np.cos(yaw)

        # state_update = [U, V, wz, wFL, wFR, wRL, wRR, yaw, x, y]
        outputs = np.array([fFLx, fFRx, fRLx, fRRx,
                            fFLy, fFRy, fRLy, fRRy,
                            fFLz, fFRz, fRLz, fRRz,
                            sFL, sFR, sRL, sRR, fFLxt, fFLyt])

        return [state_dot, vx, vy, ax, ay, outputs, axc, ayc]