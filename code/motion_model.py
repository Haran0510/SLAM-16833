'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math

def sample_normal_distribution(mean, std):

    sample = np.random.normal(mean, std)
    return sample

def wrapToPi(angle):

    return (angle + np.pi) % (2 * np.pi) - np.pi

class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.0005
        self._alpha2 = 0.0005
        self._alpha3 = 0.0005
        self._alpha4 = 0.0005


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        # return np.random.rand(3)

        # Vedang's version : 
    
        # initiliaze variables
        
        x_bar, y_bar, theta_bar = u_t0
        x_t, y_t, theta_t = u_t1
        x, y, theta = x_t0

        # calculate deltas for motion model using odometry frame - Reference from Probabilistic Robotics 
        delta_rot1 = math.atan2(y_t - y_bar, x_t - x_bar) - theta_bar
        delta_trans = math.sqrt((x_bar - x_t)**2 + (y_bar - y_t)**2)
        delta_rot2 = theta_t - theta_bar - delta_rot1

        # calculate variances for motion model using odometry frame - Reference from Probabilistic Robotics
        b1 = self._alpha1*delta_rot1**2 + self._alpha2*delta_trans**2
        b2 = self._alpha3*delta_trans**2 + self._alpha4*delta_rot1**2 + self._alpha4*delta_rot2**2
        b3 = self._alpha1*delta_rot2**2 + self._alpha2*delta_rot2**2

        # calculate true poses for motion model using - Reference from Probabilistic Robotics 
        #rot1_true = wrapToPi(delta_rot1 - sample_normal_distribution(0, math.sqrt(b1)))
        rot1_true = delta_rot1 - sample_normal_distribution(0, math.sqrt(b1))
        trans_true = delta_trans - sample_normal_distribution(0, math.sqrt(b2))
        #rot2_true = wrapToPi(delta_rot2 - sample_normal_distribution(0, math.sqrt(b3)))
        rot2_true = delta_rot2 - sample_normal_distribution(0, math.sqrt(b3))
        
        # calculate new pose using odometry frame - Reference from Probabilistic Robotics
        x_t1 = np.zeros(x_t0.shape)
        x_t1[0] = x + trans_true*math.cos(theta + rot1_true)
        x_t1[1] = y + trans_true*math.sin(theta + rot1_true)
        #x_t1[2] = wrapToPi(theta + rot1_true + rot2_true)
        x_t1[2] = theta + rot1_true + rot2_true

        return x_t1
