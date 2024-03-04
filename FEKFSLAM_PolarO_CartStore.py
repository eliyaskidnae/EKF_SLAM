from MapFeature import *
from FEKFMBL import *
from EKF_3DOFDifferentialDriveInputDisplacement import *
from conversions import *
from FEKFSLAMFeature import *
from FEKFSLAM import *

class FEKFSLAM_3DOFDDInputDisplacementMM_2DCartesianFeaturePolarObservedOM(FEKFSLAM2DCartesianFeature, FEKFSLAM, EKF_3DOFDifferentialDriveInputDisplacement):
    """
    Feature EKF Map based Localization of a 3 DOF Differential Drive Mobile Robot (:math:`x_k=[^Nx_{B_k} ~^Ny_{B_k} ~^N\\psi_{B_k} ~]^T`) using a 2D Cartesian feature map (:math:`M=[[^Nx_{F_1} ~^Ny_{F_1}] ~[x_{F_2} ~^Ny_{F_2}] ~... ~[^Nx_{F_n} ~^Ny_{F_n}]]^T`),
    and an input displacement motion model (:math:`u_k=[^B\Delta x_k~ ^B\Delta y_k ~^B\Delta z_k ~^B\Delta\psi_k]^T`). The class inherits from the following classes:
    * :class:`Cartesian2DStoredPolarObservedMapFeature`: 2D Cartesian MapFeature using the Catesian coordinates for storage and polar coordinates for landmark observations.
    * :class:`FEKFMBL`: Feature EKF Map based Localization class.
    * :class:`EKF_3DOFDifferentialDriveInputDisplacement`: EKF for 3 DOF Differential Drive Mobile Robot with input displacement motion model.
    """

    def __init__(self, *args):
        self.Feature = globals()["CartesianFeature"]
        self.Pose = globals()["Pose3D"]
        super().__init__(*args)

if __name__ == '__main__':

    M = [  CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.
    
    xs0 = np.zeros((6, 1))
    kSteps = 5000
    alpha = 0.95

    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object

    x0 = Pose3D(np.zeros((3, 1)))
    dr_robot = DR_3DOFDifferentialDrive(index, kSteps, robot, x0)
    robot.SetMap(M)

    auv = FEKFSLAM_3DOFDDInputDisplacementMM_2DCartesianFeaturePolarObservedOM(M, alpha, kSteps, robot)

    P0 = np.zeros((3,3))

    # # get Feature
    znp = np.zeros((0,1))
    Rnp = np.zeros((0,0))  # empty matrix

    zf, Rf, Hf, Vf  = auv.GetFeatures()
    Rfc = np.diag(np.array([0.01** 2,0.01** 2])) 
    for i in range(0,len(zf)):
        # reshape the feature observation and its covariance matrix
        znp = np.block([[znp], [zf[i]]])
        Rnp = scipy.linalg.block_diag(Rnp, Rfc)

    print(znp, Rnp)

    x0, P0 = auv.AddNewFeatures(x0 , P0, znp, Rnp)

    usk=np.array([[0.5, 0.03]]).T
    auv.LocalizationLoop(x0, P0, usk)

    exit(0)
