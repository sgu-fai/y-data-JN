# -*- coding: utf-8 -*-
import numpy as np
from sdc.timestamp import Timestamp

def kalman_transit_covariance(S, A, R):
    """
    :param S: Current covariance matrix
    :param A: Either transition matrix or jacobian matrix
    :param R: Current noise covariance matrix
    """
 
    return np.dot(A,np.dot(S,np.transpose(A))) + R # Assune dt = 1 


def kalman_process_observation(mu, S, observation, C, Q):
    """
    Performs processing of an observation coming from the model: z = C * x + noise
    :param mu: Current mean
    :param S: Current covariance matrix
    :param observation: Vector z
    :param C: Observation matrix
    :param Q: Noise covariance matrix (with zero mean)
    """
    
    K = np.dot(S, C.T) @ np.linalg.inv(np.dot(np.dot(C,S),C.T) + Q)   
    new_mu = mu + K @ (observation - np.dot(C,mu))
    new_S = (np.identity(len(S)) - K @ C) @ S

    return new_mu, new_S
