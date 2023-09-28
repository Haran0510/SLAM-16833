'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """

        self.occupancy_map = occupancy_map

        self._z_hit = 1
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 100

        self._sigma_hit = 50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 10  # subsampling every 10 degrees - can change value

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """

        # Getting values out of the belief vector : theta is the orientation the bot is facing 
        x, y, theta = x_t1

        theta_in_deg = theta * 180 / math.pi

        # Initializing the probability of the range scan
        prob_zt1 = 1.0

        for angle in range(-90, 90, self._subsampling):

            laser_angle_deg = theta_in_deg + angle

            laser_angle_rad = (laser_angle_deg)*(math.pi/180)

            # Original bot's laser measurement in this particular angle 

            z_t1 = z_t1_arr[angle + 90]  

            # Calculate expected measurement

            expected_zt_raycast = self.ray_casting(x, y, laser_angle_rad)

            # Calculating probabilities : 

            p_hit = self.p_hit(z_t1, expected_zt_raycast)

            p_short = self.p_short(z_t1, expected_zt_raycast)

            p_max = self.p_max(z_t1)

            p_rand = self.p_rand(z_t1)

            # Combine probabilities using sensor model parameters
            prob_zt1 *= (
                self._z_hit * p_hit +
                self._z_short * p_short +
                self._z_max * p_max +
                self._z_rand * p_rand
            )

        x_idx_particle = math.floor(x/10)
        y_idx_particle = math.floor(y/10)

        if self.occupancy_map[y_idx_particle, x_idx_particle] >= self._min_probability or self.occupancy_map[y_idx_particle, x_idx_particle] < 0 : 

            prob_zt1 = 0


        return prob_zt1

    def ray_casting(self, x, y, laser_angle_rad):
        
        # Get map size : 
        map_shape = np.shape(self.occupancy_map)

        map_size_x = map_shape[0]
        map_size_y = map_shape[1]

        # Ray-casting step size - can change : 

        raycast_step = 10 

        # Initializing ray-cast end point to starting point

        x_end_laser = x
        y_end_laser = y

        # Indices wrt the occupancy map size : 
        x_idx = int(np.round(x/10))
        y_idx = int(np.round(y/10))

        # Checking for valid indices for occupancy map matrix query : 

        while x_idx >= 0 and y_idx >= 0 and x_idx < map_size_x and y_idx < map_size_y :
                
                if self.occupancy_map[y_idx, x_idx] < self._min_probability and self.occupancy_map[y_idx, x_idx] >= 0 :

                    # Take a step in the laser direction if no obstacle was hit : 

                    x_end_laser = x_end_laser + raycast_step*np.cos(laser_angle_rad) 
                    y_end_laser = y_end_laser + raycast_step*np.sin(laser_angle_rad)
                    
                #elif self.occupancy_map[y_idx, x_idx] >= self._min_probability :

                else:

                    # Return the distance if we hit an obstacle or go out of bounds
                    return np.sqrt((x_end_laser - x)**2 + (y_end_laser - y)**2)
                
                x_idx = int(np.around(x_end_laser/10))
                y_idx = int(np.around(y_end_laser/10))

        return np.sqrt((x_end_laser - x)**2 + (y_end_laser - y)**2)
    
    # Computing the four densities 
    
    def p_hit(self, z, z_expected): 

        if z >= 0 and z <= self._max_range:
            
            denom_coeff = 1/ (self._sigma_hit * math.sqrt(2*math.pi)) 
            distbn_term = math.exp((-1/2)* ((z_expected - z)**2/self._sigma_hit**2))
            
            p_hit = denom_coeff * distbn_term

        else:

            p_hit = 0 
        
        return p_hit
    
    def p_short(self, z, z_expected):
        
        if z >= 0 and z <= z_expected:
            
            eta = 1/(1-math.exp(-self._lambda_short*z_expected))
            p_short = eta*self._lambda_short*math.exp(-self._lambda_short*z)
        
        else: 

            p_short = 0
            
        return p_short

    def p_max(self, z):
        
        if z == self._max_range: 
            
            p_max = 1.0
        
        else:

            p_max = 0.0

        return p_max

    def p_rand(self, z):
         
        if z >= 0 and z < self._max_range:
            
            p_rand = 1/(self._max_range)
        
        else:
             
            p_rand = 0.0

        return p_rand 

         
