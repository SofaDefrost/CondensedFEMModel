# -*- coding: utf-8 -*-
"""Base controller to interact with the Sofa scene.
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Jul 29 2022"

import Sofa

class BaseVisualizationController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        """Classical initialization of a python class.

        Note:
        ----
            The kwargs argument must containe:
                - root: to recover the action Controller of the scene.
                - config: link to config
        """
        assert kwargs.get("name") == "VisualizationController"
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.root =  kwargs["root"]
        self.config = kwargs["config"]
        
        self.it = 0
        self.n_eq_dt = self.config.get_n_eq_dt()
        self.n_dt = self.config.get_n_dt()
        self.post_sim_n_eq_dt = self.config.get_post_sim_n_eq_dt()

        self.n_robot = self.config.n_robot


    # def lambda_c_from_displacement(act_disp, cont_disp, xc):
    #     W = self.root.Controller.get_compliance_matrice_in_constraint_space()
    #     dfree = self.root.Controller.get_dfree()

    #     n_constraint = len(act_disp) + len(cont_disp)
    #     n_act_constraint = len(act_disp)

    #     for i in range(self.n_robot):
    #         # Get mechanical matrices from SOFA
    #         Waa_i = W_t[i * n_constraint: i * n_constraint + n_act_constraint, i * n_constraint: i * n_constraint + n_act_constraint]
    #         Wca_i = W_t[i * n_constraint + n_act_constraint :(i + 1) * n_constraint,
    #                 i * n_constraint: i * n_constraint + n_act_constraint]
    #         Wcc_i = W_t[i * n_constraint + n_act_constraint: (i + 1) * n_constraint, 
    #                 i * n_constraint + n_act_constraint: (i + 1) * n_constraint]

    #         dfree_a_i = dfree_t[i * n_constraint: i * n_constraint + n_act_constraint]
    #         dfree_c_i = dfree_t[i * n_constraint + n_act_constraint: (i + 1) * n_constraint]

    #         # Compute lambda_c from wanted actuation and contact displacements
    #         Wcc_i_inv = np.linalg.inv(Wcc_i)
    #         Waa_i_inv = np.linalg.inv(Waa_i)

    #         inter_1 = np.linalg(np.linalg(Wca_i, Waa_i_inv), np.array(act_disp) - dfree_a_i)
    #         inter_2 = np.array(cont_disp)
    #         lambda_c_i = np.linald(Wcc_i_inv, )


    def onAnimateBeginEvent(self, event):
        # Reach Equilibrium
        if self.it < self.n_eq_dt:
            n_vars = len(self.config.get_actuators_variables()) + len(self.config.get_contacts_variables())
            null_action = (n_vars*[0])*self.n_robot
            self.root.Controller.apply_actions(null_action)

        # Apply actions
        elif (self.it >= self.n_eq_dt) and (self.it < self.n_eq_dt + self.n_dt):

            if self.it == self.n_eq_dt:
                print("Equilibrium reached. Starting applying actuation")
            step = self.it - self.n_eq_dt
            init_actuation = [list(self.config.get_actuators_variables().values())[i][0] for i in range(len(list(self.config.get_actuators_variables().values())))]
            init_contact = [list(self.config.get_contacts_variables().values())[i][0] for i in range(len(list(self.config.get_contacts_variables().values())))]

            # Apply gradually actuation/contact forces
            interpolated_vars = []
            for robot in range(self.n_robot):
                interpolated_vars += [min((step + 1) * v/(self.n_dt), v) for v in init_actuation]
                interpolated_vars += [min((step + 1) * v/(self.n_dt), v) for v in init_contact]

            self.root.Controller.apply_actions(interpolated_vars)

        else:
            if self.it == self.n_eq_dt + self.n_dt:
                print("Action applied")

            if self.it == self.n_eq_dt + self.n_dt + self.post_sim_n_eq_dt:
                print("Equilibrium reached")

            init_vars = []
            for robot in range(self.n_robot):
                init_vars += [list(self.config.get_actuators_variables().values())[i][0] for i in range(len(list(self.config.get_actuators_variables().values())))]
                init_vars += [list(self.config.get_contacts_variables().values())[i][0] for i in range(len(list(self.config.get_contacts_variables().values())))]

            self.root.Controller.apply_actions(init_vars)

        self.it += 1
