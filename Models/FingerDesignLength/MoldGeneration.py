# -*- coding: utf-8 -*-
"""Mold generation for the SensorFinger"""

__authors__ = "sescaidanavarro, tnavez"
__contact__ = "stefan.escaida@uoh.cl, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Oct 28 2022"

import gmsh 
import numpy as np
import Generation
import os 


def dimTagz2Tagz(DimTagz, dimension):
    Tagz = [tag for (dim, tag) in DimTagz]
    return Tagz

def createMoldParts(config, createFinger, createCavitySketch):  

    MeshesPath = os.path.dirname(os.path.abspath(__file__))+'/Meshes/'

    def hide_all():
        ent = gmsh.model.getEntities()
        for x in ent:
            gmsh.model.setVisibility((x,), False)

    def createCableChannel(Length, Thickness):
        ChannelHeight = 1
        ChannelThickness = 2
        ChannelDepth = 3

    def createCableChannelCork(Length, Thickness):
        pass
    
    def createFingerMold(Stage1Mod=False):          
        FingerDimTag = createFinger(Stage1Mod)
        MoldBoxDimTag = (3,gmsh.model.occ.addBox(-config.ThicknessMold/2,
                                                0,
                                                config.MoldWallThickness, 
                                                config.ThicknessMold, 
                                                config.HeightMold, 
                                                -config.LengthMold))
        CableHeight = 5*config.Height/6
        CableLength = config.LengthMold+2*config.MoldWallThickness
        CableDimTag = (3,gmsh.model.occ.addCylinder(0,CableHeight,2*config.MoldWallThickness,0,0,-CableLength,config.CableRadius))
        
        #gmsh.fltk.run()
        CutOut = gmsh.model.occ.cut([MoldBoxDimTag],[FingerDimTag, CableDimTag])
        
        MoldBaseDimTag = CutOut[0][0]
        AllCavitiesDimTags = CutOut[0][1:]
        
        print("MoldBaseDimTag : ", MoldBaseDimTag )
        print("AllCavities: ", AllCavitiesDimTags)
        
        MoldBoxOuterRimDimTag = (3,gmsh.model.occ.addBox(-config.ThicknessMold/2,
                                                        0,
                                                        config.MoldWallThickness, 
                                                        config.ThicknessMold, 
                                                        -config.MoldWallThickness, 
                                                        -config.LengthMold))
        
        MoldBoxInnerRimDimTag = (3,gmsh.model.occ.addBox(-config.ThicknessMold/2+config.MoldWallThickness,
                                                        0,
                                                        0,
                                                        config.ThicknessMold-2*config.MoldWallThickness,
                                                        -config.MoldWallThickness, 
                                                        -config.LengthMold+2*config.MoldWallThickness))
        
        CutOut = gmsh.model.occ.cut([MoldBoxOuterRimDimTag],[MoldBoxInnerRimDimTag])
        MoldRim = CutOut[0][0]
        
        FuseOut = gmsh.model.occ.fuse([MoldBaseDimTag],[MoldRim])
        MoldDimTag = FuseOut[0][0]
        #MoldPG = gmsh.model.addPhysicalGroup(3,[MoldDimTag])
        
        gmsh.model.occ.synchronize()
        
        return MoldDimTag, AllCavitiesDimTags

    def createMoldLid(AllCavitiesDimTags):
        
        #-----------------
        # Create mold lid
        #-----------------
        MoldLidTopDimTag = (3,gmsh.model.occ.addBox(-config.ThicknessMold/2,
                                                    -config.MoldWallThickness,
                                                    config.MoldWallThickness, 
                                                    config.ThicknessMold, 
                                                    -config.MoldWallThickness, 
                                                    -config.LengthMold))
        gmsh.model.occ.synchronize()

    #    ROITolerance = 0.1 
        SurfaceBorderDimTags = gmsh.model.getBoundary([MoldLidTopDimTag],oriented=False)    
        print("SurfaceBorderDimTags : {}".format(SurfaceBorderDimTags))
        
        LineBorderDimTags = gmsh.model.getBoundary(SurfaceBorderDimTags[0:], combined=False, oriented=False)
        print("LineBorderDimTags : {}".format(LineBorderDimTags))
        BorderTagz = dimTagz2Tagz(LineBorderDimTags,1)

        gmsh.model.occ.fillet([MoldLidTopDimTag[1]], BorderTagz, [0.7])
        gmsh.model.occ.synchronize()
        # gmsh.fltk.run()

        MoldLidInteriorDimTag = (3,gmsh.model.occ.addBox(-config.ThicknessMold/2+config.MoldWallThickness+config.MoldCoverTolerance,
                                                        0,
                                                        config.MoldCoverTolerance, 
                                                        config.ThicknessMold-2*config.MoldWallThickness-2*config.MoldCoverTolerance,
                                                        -config.MoldWallThickness, 
                                                        -config.LengthMold+2*config.MoldWallThickness+2*config.MoldCoverTolerance))
        
        #-----------------
        # Create cavity cork
        #-----------------
        
        CorkBellowHeight = config.BellowHeight+2
        CorkWallThickness = config.WallThickness-1
        CavityCorkSketchDimTag = (2, createCavitySketch(config.OuterRadius, CorkBellowHeight, config.TeethRadius, CorkWallThickness, config.CenterThickness, config.PlateauHeight))

    #    
    #    CavityCorkSketchDimTag = (2,Generation.createCavitySketch(config.OuterRadius, config.NBellows, config.BellowHeight, config.TeethRadius, config.WallThickness/2, config.CenterThickness))
    
        ExtrudeDimTags = gmsh.model.occ.extrude([CavityCorkSketchDimTag],0,config.CavityCorkThickness,0)
        

        HalfDimTag = ExtrudeDimTags[1]
        
    #    HalfCopyDimTag = gmsh.model.occ.copy([HalfDimTag])
    #    print("HalfCopyDimTag: ", HalfCopyDimTag)
    #    gmsh.model.occ.affineTransform(HalfCopyDimTag, [1,0,0,0, 0,1,0,0, 0,0,-1,0])
    #     
    #    FusionOut = gmsh.model.occ.fuse([HalfDimTag], HalfCopyDimTag)
    #    CavityCorkDimTags = FusionOut[0]
    #    
        CavityCorkDimTags = [HalfDimTag]
        
        gmsh.model.occ.translate(CavityCorkDimTags,0,0,-config.Length-CorkBellowHeight/2)
        CavityCork2DimTags = gmsh.model.occ.copy(CavityCorkDimTags)
        CavityCork3DimTags = gmsh.model.occ.copy(CavityCorkDimTags)
        CavityCork4DimTags = gmsh.model.occ.copy(CavityCorkDimTags)    
        
        gmsh.model.occ.affineTransform(CavityCork2DimTags, [-1,0,0,0, 0,1,0,0, 0,0,1,0])
        gmsh.model.occ.translate(CavityCork3DimTags,0,0,-config.Length)
        gmsh.model.occ.affineTransform(CavityCork4DimTags, [-1,0,0,0, 0,1,0,0, 0,0,1,0])
        gmsh.model.occ.translate(CavityCork4DimTags,0,0,-config.Length)
        gmsh.model.occ.synchronize()
        AllCavitiesCorkDimTags = CavityCorkDimTags + CavityCork2DimTags + CavityCork3DimTags + CavityCork4DimTags
        
        FuseOut = gmsh.model.occ.fuse([MoldLidTopDimTag],[MoldLidInteriorDimTag]+AllCavitiesCorkDimTags+AllCavitiesDimTags)
        LidDimTag = FuseOut[0][0]
        
        
        
        return LidDimTag

        
    def creakteMoldForCork():
        
        
        CorkBellowHeight = config.BellowHeight+2
        CorkWallThickness = config.WallThickness-1
        CavityCorkSketchDimTag = (2, createCavitySketch(config.OuterRadius, CorkBellowHeight, config.TeethRadius, CorkWallThickness, config.CenterThickness, config.PlateauHeight))

        Tolerance = 2
        TotalBellowHeight = config.NBellows * config.BellowHeight
        CorkMoldHeight = TotalBellowHeight + 2 * Tolerance
        CorkMoldThickness = (config.OuterRadius+Tolerance) * 2    
        ExtrudeDimTags = gmsh.model.occ.extrude([CavityCorkSketchDimTag],0,config.CavityCorkThickness,0)    
        HalfDimTag = ExtrudeDimTags[1]
            
    #    HalfCopyDimTag = gmsh.model.occ.copy([HalfDimTag])
    #    print("HalfCopyDimTag: ", HalfCopyDimTag)
    #    gmsh.model.occ.affineTransform(HalfCopyDimTag, [1,0,0,0, 0,1,0,0, 0,0,-1,0])
    #     
    #    FusionOut = gmsh.model.occ.fuse([HalfDimTag], HalfCopyDimTag)
    #    CavityCorkDimTag = FusionOut[0][0]
        CavityCorkDimTag = HalfDimTag
        gmsh.model.occ.translate([CavityCorkDimTag],0,0,-TotalBellowHeight/2)
        
        CavityCork2DimTags = gmsh.model.occ.copy([CavityCorkDimTag])
        gmsh.model.occ.affineTransform(CavityCork2DimTags, [-1,0,0,0, 0,1,0,0, 0,0,1,0])
        
        FuseOut = gmsh.model.occ.fuse([CavityCorkDimTag],CavityCork2DimTags)
        
        CompleteCorkDimTag = FuseOut[0][0]
        CavityMoldBoxDimTag = (3,gmsh.model.occ.addBox(-CorkMoldThickness/2,
                                                    -Tolerance,
                                                    -CorkMoldHeight/2,
                                                    CorkMoldThickness, 
                                                    config.CavityCorkThickness+Tolerance,
                                                    CorkMoldHeight+Tolerance))
        CutOut = gmsh.model.occ.cut([CavityMoldBoxDimTag],[CompleteCorkDimTag])
        gmsh.model.occ.synchronize()
        gmsh.write(MeshesPath + "MoldForCork.step")
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.refine()
        gmsh.write(MeshesPath + "MoldForCork.stl")
        
        
    def createFingerClamp():
        
    #    gmsh.merge("Finger_Parametric.step")
    #    FingerDimTags = gmsh.model.getEntities(3)
        FingerDimTags = [createFinger()]
        
        ClampBoxWidth = 4*config.FixationWidth+config.Thickness
        ClampBoxLength = 4*config.FixationWidth
        ClampBoxHeight = config.Height + 2 * config.FixationWidth 
        ClampBoxDimDag = (3,gmsh.model.occ.addBox(-ClampBoxWidth/2,
                                            0,
                                            -ClampBoxLength/2,
                                            ClampBoxWidth,
                                            ClampBoxHeight,
                                            ClampBoxLength))
        ClampBoxCablePassDimTag = (3,gmsh.model.occ.addBox(-config.Thickness/2, 
                                                        0,
                                                        0,
                                                        config.Thickness,
                                                        5,
                                                        10
                                                        ))
        
        ScrewRadius = 1.7
        ScrewEarWidth = 6
        ScrewEarHeight = 3 
        ScrewEarLength = ScrewEarWidth
        
        ScrewEarBoxDimDag = (3,gmsh.model.occ.addBox(ClampBoxWidth/2,
                                            0,
                                            -ScrewEarLength/2,
                                            ScrewEarWidth,
                                            ScrewEarHeight,
                                            ScrewEarLength))
        
        ScrewLength = 6
        ScrewCylinderDimTag = (3,gmsh.model.occ.addCylinder(ClampBoxWidth/2+ScrewEarWidth/2,-ScrewLength/3,0, 0,ScrewLength,0,ScrewRadius))
        
        
        ScrewEarBox2DimTags = gmsh.model.occ.copy([ScrewEarBoxDimDag])
        gmsh.model.occ.affineTransform(ScrewEarBox2DimTags, [-1,0,0,0, 0,1,0,0, 0,0,1,0])
        
        ScrewCylinder2DimTags = gmsh.model.occ.copy([ScrewCylinderDimTag])
        gmsh.model.occ.affineTransform(ScrewCylinder2DimTags, [-1,0,0,0, 0,1,0,0, 0,0,1,0])
        
        CableHeight = 5*config.Height/6
        CableLength = config.LengthMold+2*config.MoldWallThickness
        CableDimTag = (3,gmsh.model.occ.addCylinder(0,CableHeight,2*config.MoldWallThickness,0,0,-CableLength,config.CableRadius))
        
        FuseOut = gmsh.model.occ.fuse([ClampBoxDimDag],[ScrewEarBoxDimDag]+ScrewEarBox2DimTags)
        PositiveBoxDimTag = FuseOut[0][0]
        
        gmsh.model.occ.cut([PositiveBoxDimTag],FingerDimTags+ScrewCylinder2DimTags + [ScrewCylinderDimTag,CableDimTag]+[ClampBoxCablePassDimTag])
        gmsh.model.occ.synchronize()
        gmsh.write(MeshesPath + "FingerClamp.step")
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.refine()
        gmsh.write(MeshesPath + "FingerClamp.stl")    

    MoldDimTag, AllCavitiesDimTags = createFingerMold(Stage1Mod=False)
    LidDimTag = createMoldLid(AllCavitiesDimTags)
    hide_all()
    gmsh.model.setVisibility((LidDimTag,),False, True)
    gmsh.model.setVisibility((MoldDimTag,),False, True)    
    print("MoldDimTag: ", MoldDimTag)
    print("LiddDimTag: ", LidDimTag)
    gmsh.write(MeshesPath + "Mold.step")    
    gmsh.model.occ.synchronize()    

    gmsh.clear()
    creakteMoldForCork()
    gmsh.clear()
    createFingerClamp()
    gmsh.model.occ.synchronize()

    return MoldDimTag, LidDimTag
