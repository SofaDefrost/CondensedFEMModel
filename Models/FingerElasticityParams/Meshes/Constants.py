# -*- coding: utf-8 -*-
"""Constants for the generation of the mesh.
"""

__authors__ = "emenager, tnavez, sescaidanavarro"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Jun 29 2022"

import numpy as np

# Geometric parameters
Length = 40
Height = 20
JointHeight = 6
Thickness = 17.5
JointSlopeAngle = np.deg2rad(45)
FixationWidth = 3

OuterRadius = Thickness/2 + 6
NBellowSteps = 1
StepHeight = 4
TeethRadius = Thickness/2
WallThickness = 3.5
CenterThickness = 1.5
CavityCorkThickness = 4

# Elasticity parameters
PoissonRation = 0.3
YoungsModulus = 3000

# Mold parameters
MoldWallThickness = 3
MoldCoverTolerance = 0.2
LengthMold = 3*Length + 2*MoldWallThickness
LidHoleBorderThickness = 1
LidHoleThickness = Thickness - 2*LidHoleBorderThickness
LidHoleLength = 3*Length/5

MoldHoleThickness = Thickness - 2*LidHoleBorderThickness
MoldHoleLength = Length/2

ThicknessMold = 2*OuterRadius + 2*MoldWallThickness
LengthMold = 3*Length + 2*MoldWallThickness
HeightMold = Height + FixationWidth + MoldWallThickness
MoldHoleLidBorderThickness = 2

#Cable
CableRadius = 0.75
CableDistance = 10
