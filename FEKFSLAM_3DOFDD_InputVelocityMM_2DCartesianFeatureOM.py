from FEKFSLAM import *
from FEKFMBL import *
from EKF_3DOFDifferentialDriveInputDisplacement import *
from Pose import *
from blockarray import *
from MapFeature import *
import numpy as np
from FEKFSLAMFeature import *

class FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM(FEKFSLAM2DCartesianFeature, FEKFSLAM, EKF_3DOFDifferentialDriveInputDisplacement):
    def __init__(self, *args):

        self.Feature = globals()["CartesianFeature"]
        self.Pose = globals()["Pose3D"]
       
        super().__init__(*args)


    # def GetFeatures(self):
    # Get features is inherited from EKF_3DOFDifferentialDriveInputDisplacement

if __name__ == '__main__':

    M = [  
           CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T),


        ]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6, 1))
    kSteps = 5000
    alpha = 0.99

    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object

    x0 = Pose3D(np.zeros((3, 1)))
    robot.SetMap(M)

    auv = FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM([], alpha, kSteps, robot)

    P0 = np.zeros((3,3))
    usk=np.array([[0.5,0.03]]).T

    # # get Feature
    znp = np.zeros((0,1))
    Rnp = np.zeros((0,0))  # empty matrix
    zf, Rf, Hf, Vf  = auv.GetFeatures()
    
    for i in range(0,len(zf)):
        # reshape the feature observation and its covariance matrix
        znp = np.block([[znp], [zf[i]]])
        Rnp = scipy.linalg.block_diag(Rnp, Rf[i])

    # Add Feture to the map initially 
    if(len(znp) > 0):
         x0, P0 = auv.AddNewFeatures(x0 , P0, znp, Rnp)
    auv.LocalizationLoop(x0, P0, usk)

    exit(0)
