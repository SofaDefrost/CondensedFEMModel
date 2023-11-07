# -*- coding: utf-8 -*-
"""Base controller to interact with the Sofa scene.
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Jul 19 2022"

import Sofa
import Sofa.CondensedFEMModel

class BaseController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        """Classical initialization of a python class.

        Note:
        ----
            The kwargs argument must containe:
                - constraint_solver: to recover the matrice W (compliance matrice in constraint space).
                - list_actuators: to indicate what are the actuators.
                - list_contacts: to indicate what are the actuators.
                    Note: the contact are used as actuation during the acquisition of the data.
        """
        assert kwargs.get("name") == "Controller"
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.constraint_solver = kwargs.get("constraint_solver")
        self.constraint_solver_setter = kwargs.get("constraint_solver_setter")
        self.list_actuators = kwargs.get("list_actuators")
        self.list_effectors = kwargs.get("list_effectors")
        self.list_effectors_MO = kwargs.get("list_effectors_MO")
        self.list_contacts = None

    def get_actuators(self):
        """List all the actuators in the scene.
        Contact Points are handled as actuators in the scene.

        Outputs:
        --------
            list_actuators: list of Sofa Object
                The list of all actuators in the scene.

        Note:
        ----
            The actuators are given in the initialization steps.
        """
        assert len(self.list_actuators)!= 0
        return self.list_actuators

    def get_effectors(self):
        """List all the effectors in the scene.

        Outputs:
        --------
            list_effectors: list of Sofa Object
                The list of all effectors in the scene.

        Note:
        ----
            The effectors are given in the initialization steps.
        """
        return self.list_effectors

    def get_n_effectors_constraints(self):
        """
        Get the number of constraints impose by the effectors.

        Outputs:
        -------
            The number of constraints impose by the effectors.
        """
        n_constraint = 0
        for effector in self.list_effectors:
            n_constraint+= effector.useDirections.value[:].sum()
        return n_constraint

    def get_compliance_matrice_in_constraint_space(self):
        """Recover the matrice W from the scene.

        Outputs:
        -------
            W: array
                The compliance matrice in constraint space.
        """
        return self.constraint_solver.W()

    def get_lambda(self):
        """Recover the vector lambda from the scene.

        lambda is the vector of constraints.

        Outputs:
        -------
            lambda: array
                Vector of constraints.
        """
        return self.constraint_solver.lambda_force()

    def get_dfree(self):
        """Recover the vector dfree from the scene.

        dfree is displacement in free configuration without any actuation.

        Outputs:
        -------
            dfree: array
                Displacement in free configuration without any actuation.
        """
        return self.constraint_solver.dfree()

    @staticmethod
    def get_actuators_state(self):
        """Return the actual state of the actuators (volume, lentgh).

        This function depends of the Sofa scene, so we have to implement it in
        the scene.
        """
        return None

    @staticmethod
    def get_effectors_state(self):
        """Return the actual state of the effectors.

        This function depends of the Sofa scene, so we have to implement it in
        the scene.
        """
        return None

    @staticmethod
    def get_effectors_positions(self):
        """Return the positions of the effectors.

        This function depends of the Sofa scene, so we have to implement it in
        the scene.
        """
        return None

    @staticmethod
    def apply_actions(self, values):
        """Apply the actuation in the scene.

        Parameters:
        -----------
            values: list of float
                The values of the actuators (in the same order as the list_actuators).
        """
        return None

    def get_contacts(self):
        """Recover the contact in the scene.

        """
        return None


    def onAnimateEndEvent(self, event):
        print_compliance = False
        if print_compliance:
            W_0 = self.get_compliance_matrice_in_constraint_space()
            dfree_0 = self.get_dfree()
            print("W_0:", W_0, "dfree_0:", dfree_0)
