# -*- coding: utf-8 -*-
"""Base controller to interact with the Sofa scene.
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Aug 2 2023"

# import Sofa
import utils
import numpy as np

import polhemus_liberty.python.PolhemusUSB as tracking
import phidget1002_0b.Multiple_Actuators_Class as actuation


class PID():
    def __init__(self, P, I, D):
        self.P = P
        self.I = I
        self.D = D

        self.pred_error = 0
        self.corr_integral = 0

    def reinit(self):

        self.pred_error = 0
        self.corr_integral = 0

    def compute_correction(self, real_pos, goal):

        print("-----------")
        print(real_pos)
        print(goal)

        error = goal - real_pos

        corr_prop = self.P * error
        self.corr_integral = self.corr_integral + self.I * error
        corr_deriv = self.D * (error - self.pred_error)

        self.pred_error = error

        return goal + corr_prop + self.corr_integral + corr_deriv


class EmbeddedController():
    def __init__(self, *args, **kwargs):
        """Classical initialization of a python class.

        Note:
        ----
            The kwargs argument must containe:
                - root: the root of the SOFA scene.
                - actuators: the actuators we want to control (force)
        """
        super(EmbeddedController, self).__init__(*args, **kwargs)

        # self.root = kwargs.get("root")
        # self.list_actuators = kwargs.get("list_actuators")
        # self.goalMO = kwargs.get("goalMO")
        # self.effectorMO = kwargs.get("effectorMO")

        self.n_act_constraint = 6
        self.W0, self.dfree0, self.scaling, self.model, self.n_constraint = utils.create_network()
        self.problem = utils.init_QP(self.n_act_constraint)

        self.delta_a = None
        self.s_a = None

        self.step = 0

        self.waiting_time = 1
        self.step_per_goal = 20

        #Create trajectory
        self.get_trajectory()
        self.goal = self.goals[0]

        #Create the PID for the correction
        self.PID = PID(P=0.2, I=0.1, D=0.)

        #Connect the hardware :
        self.tracker = tracking.PolhemusOvercoat(init_pos = [100,0,0]) 
        self.actuators = actuation.Multiple_Actuators()

        
    def get_trajectory(self):
        """Generate the trajectory we want to apply in the scene.

        Note:
        ----
            self.goals: list of 3D goals

        """
        self.goals = []

        # circle_center = np.array([110, 0, 0])
        # circle_radius = 30
        # n_samples = 20
        # for i in range(n_samples):
        #    self.goals.append(circle_center + np.array([0, circle_radius * np.cos(2 * np.pi * i / n_samples), 0]))
        
        
        # 3D spiral trajectory
        center= np.array([110 - 0.25, 0, 0])
        radius = 25
        n_samples = 20
        height_increment = 0.5 / n_samples
        additional_height = 0.0
        angle_increment = 2 * 6.28319 / n_samples
        additional_angle = 0.0
        
        for i in range(n_samples):
            x = center[0] + additional_height
            y = center[1] + radius * np.cos(additional_angle)
            z = center[2] + radius * np.sin(additional_angle)
            self.goals.append(np.array([x,y,z]))

            additional_height += height_increment
            additional_angle += angle_increment
            
    # def apply_lambda(self, lambda_a):
    #     for i, actuator in enumerate(self.list_actuators):
    #         actuator.value.value = [lambda_a[i]]

    def EachTimeStep(self):
        
        if self.step == self.waiting_time:
            #The first state is given by applying no force

            Waa0 =self.W0[:self.n_act_constraint, :self.n_act_constraint]
            dfreea0 = self.dfree0[:self.n_act_constraint]
            self.s_a = (utils.compute_delta_a(np.array([0 for _ in range(self.n_act_constraint)]), Waa0, dfreea0)).tolist()

        if self.step >= self.waiting_time:

            #Recover the true position of the effector
            [pos, q] = self.tracker.get_data()
            # pos = self.effectorMO.position.value[0]

            #Correct the goal to take into account the difference between model and reality
            corrected_goal = self.PID.compute_correction(pos, self.goal)

            #Predict mechanical matrices from the network
            Waa, Wea, dfree_a, dfree_e = utils.predict_W_dFree(self.model, self.W0, self.dfree0, self.s_a, self.scaling, self.n_constraint,
                                                               self.n_act_constraint, corrected_goal, nb_effector=1)

            #Compute optimization problem's matrices from the mechanical matrices
            H, g, A, lb, ub, lbA, ubA = utils.build_QP_system(Waa, Wea, dfree_a, dfree_e, use_epsilon=False)

            #Solve the optimization problem to recover lambda_a, the force to apply
            lambda_a = utils.solve_QP(self.problem, H, g, A, lb, ub, lbA, ubA, is_init=(self.step != self.waiting_time))

            #Compute the new actuators' state from mechanical matrices and forces
            self.s_a = (utils.compute_delta_a(lambda_a, Waa, dfree_a)).tolist()

            #Apply the forces
            self.actuators.apply_pressure_tab(pressure_tab = lambda_a)

            # self.apply_lambda(lambda_a)


        self.step+=1

        #Manage the trajectory
        self.goal = self.goals[min(self.step//self.step_per_goal, len(self.goals)-1)]
        # self.goalMO.position.value = [self.goal]
