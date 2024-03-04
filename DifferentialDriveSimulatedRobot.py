from SimulatedRobot import *
from IndexStruct import *
from Pose import *
from Feature import *
from MapFeature import *
from conversions import *
import scipy
from roboticstoolbox.mobile.Animations import *
import numpy as np


class DifferentialDriveSimulatedRobot(SimulatedRobot):
    """
    This class implements a simulated differential drive robot. It inherits from the :class:`SimulatedRobot` class and
    overrides some of its methods to define the differential drive robot motion model.
    """
    def __init__(self, xs0, map=[],*args):
        """
        :param xs0: initial simulated robot state :math:`\\mathbf{x_{s_0}}=[^Nx{_{s_0}}~^Ny{_{s_0}}~^N\psi{_{s_0}}~]^T` used to initialize the  motion model
        :param map: feature map of the environment :math:`M=[^Nx_{F_1},...,^Nx_{F_{nf}}]`

        Initializes the simulated differential drive robot. Overrides some of the object attributes of the parent class :class:`SimulatedRobot` to define the differential drive robot motion model:

        * **Qsk** : Object attribute containing Covariance of the simulation motion model noise.

        .. math::
            Q_k=\\begin{bmatrix}\\sigma_{\\dot u}^2 & 0 & 0\\\\
            0 & \\sigma_{\\dot v}^2 & 0 \\\\
            0 & 0 & \\sigma_{\\dot r}^2 \\\\
            \\end{bmatrix}
            :label: eq:Qsk

        * **usk** : Object attribute containing the simulated input to the motion model containing the forward velocity :math:`u_k` and the angular velocity :math:`r_k`

        .. math::
            \\bf{u_k}=\\begin{bmatrix}u_k & r_k\\end{bmatrix}^T
            :label: eq:usk

        * **xsk** : Object attribute containing the current simulated robot state

        .. math::
            x_k=\\begin{bmatrix}{^N}x_k & {^N}y_k & {^N}\\theta_k & {^B}u_k & {^B}v_k & {^B}r_k\\end{bmatrix}^T
            :label: eq:xsk

        where :math:`{^N}x_k`, :math:`{^N}y_k` and :math:`{^N}\\theta_k` are the robot position and orientation in the world N-Frame, and :math:`{^B}u_k`, :math:`{^B}v_k` and :math:`{^B}r_k` are the robot linear and angular velocities in the robot B-Frame.

        * **zsk** : Object attribute containing :math:`z_{s_k}=[n_L~n_R]^T` observation vector containing number of pulses read from the left and right wheel encoders.
        * **Rsk** : Object attribute containing :math:`R_{s_k}=diag(\\sigma_L^2,\\sigma_R^2)` covariance matrix of the noise of the read pulses`.
        * **wheelBase** : Object attribute containing the distance between the wheels of the robot (:math:`w=0.5` m)
        * **wheelRadius** : Object attribute containing the radius of the wheels of the robot (:math:`R=0.1` m)
        * **pulses_x_wheelTurn** : Object attribute containing the number of pulses per wheel turn (:math:`pulseXwheelTurn=1024` pulses)
        * **Polar2D_max_range** : Object attribute containing the maximum Polar2D range (:math:`Polar2D_max_range=50` m) at which the robot can detect features.
        * **Polar2D\_feature\_reading\_frequency** : Object attribute containing the frequency of Polar2D feature readings (50 tics -sample times-)
        * **Rfp** : Object attribute containing the covariance of the simulated Polar2D feature noise (:math:`R_{fp}=diag(\\sigma_{\\rho}^2,\\sigma_{\\phi}^2)`)

        Check the parent class :class:`prpy.SimulatedRobot` to know the rest of the object attributes.
        """
        super().__init__(xs0, map,*args) # call the parent class constructor

        # Initialize the motion model noise
        self.Qsk = np.diag(np.array([0.1 ** 2, 0.01 ** 2, np.deg2rad(1) ** 2]))  # simulated acceleration noise
        self.usk = np.zeros((3, 1))  # simulated input to the motion model

        # Inititalize the robot parameters
        self.wheelBase = 0.5  # distance between the wheels
        self.wheelRadius = 0.1  # radius of the wheels
        self.pulse_x_wheelTurns = 10000  # number of pulses per wheel turn

        # Initialize the sensor simulation
        self.encoder_reading_frequency = 1  # frequency of encoder readings
        self.Re= np.diag(np.array([22 ** 2, 22 ** 2]))  # covariance of simulated wheel encoder noise

        self.Cartesian2D_feature_reading_frequency = 50 # frequency of Polar2D feature readings
        self.Cartesian2D_max_range = 50  # maximum Cartesian2Drange, used to simulate the field of view
        self.Rfc = np.diag(np.array([0.05 ** 2,0.01** 2]))  # covariance of simulated Polar2D feature noise

        self.Polar2D_feature_reading_frequency = 50  # frequency of Polar2D feature readings
        self.Polar2D_max_range = 50  # maximum Polar2D range, used to simulate the field of view
        self.Rfp = np.diag(np.array([1 ** 2, np.deg2rad(1.5) ** 2]))  # covariance of simulated Polar2D feature noise
        
        self.xy_feature_reading_frequency = 1 # frequency of XY feature readings
        self.xy_max_range = 50  # maximum XY range, used to simulate the field of view
        self.yaw_reading_frequency = 100# frequency of Yasw readings
        self.v_yaw_std = np.deg2rad(5)  # std deviation of simulated heading noise

    def fs(self, xsk_1, usk):  # input velocity motion model with velocity noise
        """ Motion model used to simulate the robot motion. Computes the current robot state :math:`x_k` given the previous robot state :math:`x_{k-1}` and the input :math:`u_k`:

        .. math::
            \\eta_{s_{k-1}}&=\\begin{bmatrix}x_{s_{k-1}} & y_{s_{k-1}} & \\theta_{s_{k-1}}\\end{bmatrix}^T\\\\
            \\nu_{s_{k-1}}&=\\begin{bmatrix} u_{s_{k-1}} &  v_{s_{k-1}} & r_{s_{k-1}}\\end{bmatrix}^T\\\\
            x_{s_{k-1}}&=\\begin{bmatrix}\\eta_{s_{k-1}}^T & \\nu_{s_{k-1}}^T\\end{bmatrix}^T\\\\
            u_{s_k}&=\\nu_{d}=\\begin{bmatrix} u_d& r_d\\end{bmatrix}^T\\\\
            w_{s_k}&=\\dot \\nu_{s_k}\\\\
            x_{s_k}&=f_s(x_{s_{k-1}},u_{s_k},w_{s_k}) \\\\
            &=\\begin{bmatrix}
            \\eta_{s_{k-1}} \\oplus (\\nu_{s_{k-1}}\\Delta t + \\frac{1}{2} w_{s_k}) \\\\
            \\nu_{s_{k-1}}+K(\\nu_{d}-\\nu_{s_{k-1}}) + w_{s_k} \\Delta t
            \\end{bmatrix} \\quad;\\quad K=diag(k_1,k_2,k_3) \\quad k_i>0\\\\
            :label: eq:fs

        Where :math:`\\eta_{s_{k-1}}` is the previous 3 DOF robot pose (x,y,yaw) and :math:`\\nu_{s_{k-1}}` is the previous robot velocity (velocity in the direction of x and y B-Frame axis of the robot and the angular velocity).
        :math:`u_{s_k}` is the input to the motion model contaning the desired robot velocity in the x direction (:math:`u_d`) and the desired angular velocity around the z axis (:math:`r_d`).
        :math:`w_{s_k}` is the motion model noise representing an acceleration perturbation in the robot axis. The :math:`w_{s_k}` acceleration is the responsible for the slight velocity variation in the simulated robot motion.
        :math:`K` is a diagonal matrix containing the gains used to drive the simulated velocity towards the desired input velocity.

        Finally, the class updates the object attributes :math:`xsk`, :math:`xsk\_1` and  :math:`usk` to made them available for plotting purposes.

        **To be completed by the student**.

        :parameter xsk_1: previous robot state :math:`x_{s_{k-1}}=\\begin{bmatrix}\\eta_{s_{k-1}}^T & \\nu_{s_{k-1}}^T\\end{bmatrix}^T`
        :parameter usk: model input :math:`u_{s_k}=\\nu_{d}=\\begin{bmatrix} u_d& r_d\\end{bmatrix}^T`
        :return: current robot state :math:`x_{s_k}`
        """

        # TODO: to be completed by the student
        K = np.diag([1,1,1]) # 3x3

        wsk = np.random.multivariate_normal([0,0,0],self.Qsk).reshape(3,1) # 3x1
        # previous pose and velocity
        prev_pose = Pose3D(np.array([[xsk_1[0, 0]], [xsk_1[1, 0]], [xsk_1[2, 0]]]))
        prev_vel = Pose3D(np.array([[xsk_1[3, 0]], [xsk_1[4, 0]], [xsk_1[5, 0]]]))
        # desired velocity
        desired_vel = Pose3D(np.array([[usk[0, 0]], [0] ,  [usk[1, 0]]]))#  3x1
        # current pose and velocity
        current_pose= prev_pose.oplus(Pose3D(np.array(prev_vel * self.dt + 0.5 * wsk * self.dt*self.dt))) # 3x1     
        current_velocity = prev_vel + K@(desired_vel - prev_vel) + wsk * self.dt # 3x1       
        # current robot state
        self.usk = desired_vel
        
        # current robot state
        self.xsk = np.concatenate((current_pose, current_velocity))
        self.xsk_1 = xsk_1
        if self.k % self.visualizationInterval == 0:
                self.PlotRobot()
                self.xTraj.append(self.xsk[0, 0])
                self.yTraj.append(self.xsk[1, 0])
                self.trajectory.pop(0).remove()
                self.trajectory = plt.plot(self.xTraj, self.yTraj, marker='.', color='orange', markersize=1)

        self.k += 1 #update the time step
        return self.xsk 

    def ReadEncoders(self):
        """ Simulates the robot measurements of the left and right wheel encoders.

        **To be completed by the student**.

        :return zsk,Rsk: :math:`zk=[n_L~n_R]^T` observation vector containing number of pulses read from the left and right wheel encoders. :math:`R_{s_k}=diag(\\sigma_L^2,\\sigma_R^2)` covariance matrix of the read pulses.
        """     
        # TODO: to be completed by the student
        # Get the linear and angular velocities in the robot B-Frame
        # TODO: to be completed by the student
        #distances traveled by the left and right wheels
        dl = self.usk[0, 0] * self.dt - self.wheelBase * self.usk[2, 0] * self.dt * 0.5
        dr = self.usk[0, 0] * self.dt + self.wheelBase * self.usk[2, 0] * self.dt * 0.5
        # number of pulses read from the left and right wheel encoders
        nL = dl * self.pulse_x_wheelTurns / (2 * np.pi * self.wheelRadius)
        nR = dr * self.pulse_x_wheelTurns / (2 * np.pi * self.wheelRadius)
        noise = np.random.normal(0, np.sqrt(self.Re.diagonal())).reshape((len(self.Re),1)).astype(int)
        zsk = np.array([[nL],[nR]])   + noise     
        Rsk = self.Re      
        if self.k % self.encoder_reading_frequency == 0:
            return zsk, Rsk
        else:
            return np.zeros((0,0)), np.zeros((0,0))

    def ReadCompass(self):
        """ Simulates the compass reading of the robot.
        :return: yaw and the covariance of its noise *R_yaw*
        """  
        # TODO: to be completed by the student  
        yaw = np.array([])
        R_yaw = []
    
        if(self.k % self.yaw_reading_frequency == 0):
            R_yaw = np.diag([self.v_yaw_std**2])  
            noise = np.random.normal(0, np.sqrt(R_yaw.diagonal())).reshape((len(R_yaw),1))
            # get yaw reading
            
            # compute covariance 
            yaw = np.array([self.xsk[2,0]]).reshape(1,1) + noise
                
        return yaw , R_yaw
    
    def ReadCartesian2DFeature(self):
        """ Simulates the feature reading of the robot.

        :return: 2D Cartesian distance from the robot to features in the robot frame and the covariance of its noise *R_feature*
        """
        # TODO: to be completed by the student
        zk = []
        Rk = []  
        for NxF in self.M:
            # get feature and compute feature observation with some noise
            # noise = np.random.normal(0.00001,0.00001,(2,1))
            x_f,y_f = NxF ## Feature Pose 
            x_r,y_r = self.xsk[0,0] , self.xsk[1,0] # Robot pose
            distance = sqrt((x_f-x_r)**2 + (y_f-y_r)**2) # calculate distance range
            noise = np.random.normal(0, np.sqrt(self.Rfc.diagonal())).reshape((len(self.Rfc),1))
            if(distance < self.Cartesian2D_max_range):
                 zk.append(NxF.boxplus(Pose3D.ominus( self.xsk[0:3,0].reshape((3,1)))) + noise)           
                # Measurement noise
                 Rk.append(self.Rfc)  
            
        if self.k % self.Cartesian2D_feature_reading_frequency == 0:
        
            return zk, Rk
        else:
            return [], []  
    def ReadPolar2DFeature(self):
        """ Simulates the feature reading of the robot.

        :return: 2D Cartesian distance from the robot to features in the robot frame and the covariance of its noise *R_feature*
        """
        # TODO: to be completed by the student
        zk = []
        Rk = []
        for NxF in self.M:
            # get feature 
            noise = np.random.normal(0, np.sqrt(self.Rfp.diagonal())).reshape((len(self.Rfp),1))
            x_f,y_f = NxF ## Feature Pose 
            x_r,y_r = self.xsk[0,0] , self.xsk[1,0] # Robot pose
            distance = sqrt((x_f-x_r)**2 + (y_f-y_r)**2) # calculate distance range
            
            if(distance < self.Polar2D_max_range): # only features in the distance range to be valid
                # add noise to the feature observation
                Cartes_Feature = NxF.boxplus(Pose3D.ominus(self.xsk[0:3,0].reshape((3,1)))) + noise
                Polar_Feature = c2p(Cartes_Feature)
                zk.append(Polar_Feature)
                
                # Measurement noise
                Rk.append(self.Rfp)

        if self.k % self.Polar2D_feature_reading_frequency == 0:
            return zk, Rk
        else:
            return [], []
                     
    ## This Polar Feature Observed and Stored  in Polar 
    def ReadPolar2DFeatureAll(self):
        """ Simulates the feature reading of the robot.

        :return: 2D Cartesian distance from the robot to features in the robot frame and the covariance of its noise *R_feature*
        """

        # TODO: to be completed by the student

        zk = []
        Rk = []   
    
        for NxF in self.M:
            # get feature and add some noise
            noise = np.random.normal(0,0.001,(3,1))  
            NxF = p2c(NxF) 
            Cartes_Feature = NxF.boxplus(Pose3D.ominus(self.xsk[0:3,0].reshape((3,1)) + noise))
            Polar_Feature = c2p(Cartes_Feature )
            zk.append(Polar_Feature)
            # Measurement noise
            Rk.append(self.Rfp)

        if self.k % self.Polar2D_feature_reading_frequency == 0:
            return zk, Rk
        else:
            return [], []
      
    def PlotRobot(self):
        """ Updates the plot of the robot at the current pose """

        self.vehicleIcon.update([self.xsk[0], self.xsk[1], self.xsk[2]])
        plt.pause(0.0000001)
        return