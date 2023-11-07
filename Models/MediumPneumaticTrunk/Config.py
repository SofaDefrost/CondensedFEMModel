# -*- coding: utf-8 -*-
"""Config for the AbstractPneumaticTrunk.py
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "March 23 2023"

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()) + "/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from AbstractPneumaticTrunk.AbstractPneumaticTrunkConfig import AbstractPneumaticTrunkConfig

class Config(AbstractPneumaticTrunkConfig):
    def __init__(self):
        super(Config, self).__init__("MediumPneumaticTrunk")
