# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import copy
import sys
import matplotlib.pyplot as plt
import math
import os

'''Gaussian Mixture Models (GMMs) for color detection'''

class GMM:

    def __init__(self):
        self.bound = 0.0001
        self.max_iter = 500
        self.K = 4                          # Number of Gaussians
        self.gamma = [1./self.K]*self.K         
        self.log_likelihoods = []
        self.log_likelihood = 0
        
        def merge_data():
            merged = []
            script_dir = os.path.dirname(os.path.abspath(__file__))
            picturesFolderPath = script_dir +'/SampleShades'
            background_list = ["SampleShades/" + f for f in os.listdir(picturesFolderPath)]
            for filename in background_list:
                img = cv2.imread(filename)
                nx, ny, ch = img.shape
                img = np.reshape(img, (nx*ny,ch))
                for i in range(nx):
                    merged.append(img[i, :])
            return np.array(merged)

        self.data = merge_data()
        self.n_features = self.data.shape[0]
        self.n_observations = self.data.shape[1]
        self.cluster_prob = np.ndarray([self.n_features,self.K],np.float64)

        def init_params(self):
            mean = np.array([self.data[np.random.choice(self.n_features, 1)]], np.float64)
            cov = [np.random.randint(1,255)*np.eye(self.n_observations)]
            cov = np.matrix(np.multiply(cov,np.random.rand(self.n_observations, self.n_observations)))
            return {'mean': mean, 'cov': cov}

        self.parameters = [init_params(self) for cluster in range (self.K)]

    def gaussian(self, d, mean, cov):
        det_cov = np.linalg.det(cov)
        cov_inv = np.zeros_like(cov)
        diff = np.matrix(d - mean)
        for i in range(self.n_observations):
            cov_inv[i, i] = 1/cov[i, i] 
        N = (2.0 * np.pi) ** (-len(self.data[1]) / 2.0) * (1.0 / (np.linalg.det(cov) ** 0.5)) * np.exp(-0.5 * np.sum(np.multiply(diff*cov_inv, diff), axis=1))
        return N
            
    def expectation(self):
        cluster_prob = np.ndarray([self.n_features, self.K], np.float64)
        for cluster in range(self.K):
            cluster_prob[:, cluster:cluster+1] = self.gaussian(self.data, self.parameters[cluster]['mean'], self.parameters[cluster]['cov'])*self.gamma[cluster]
        return cluster_prob

    def maximization(self, cluster_prob):
        
        cluster_sum = np.sum(cluster_prob, axis=1)
        self.log_likelihood = np.sum(np.log(cluster_sum))
        self.log_likelihoods.append(self.log_likelihood)
        cluster_prob = np.divide(cluster_prob, np.tile(cluster_sum,(self.K,1)).T)
        Nk = np.sum(cluster_prob, axis = 0)

        for cluster in range(self.K):
            temp_sum = math.fsum(cluster_prob[:, cluster])
            new_mean = 1./ Nk[cluster]* np.sum(cluster_prob[:, cluster]*self.data.T, axis=1).T       
            self.parameters[cluster]['mean'] = new_mean
            diff = self.data - self.parameters[cluster]['mean']
            new_cov = np.array(1./ Nk[cluster]*np.dot(np.multiply(diff.T, cluster_prob[:, cluster]), diff)) 
            self.parameters[cluster]['cov'] = new_cov
            self.gamma[cluster] = 1./ self.n_features * Nk[cluster]
        
        return self.gamma, self.parameters

    def em_iteration(self):
        for i in range(self.max_iter):
            mix_c, params = self.maximization(self.expectation())

            if len(self.log_likelihoods)<2: continue
            if np.abs(self.log_likelihood-self.log_likelihoods[-2])<self.bound : break
        return mix_c, params

    def save_weights(self):
        g, p = self.maximization(self.expectation())
        np.save('weights_g.npy', g, allow_pickle=True)
        np.save('parameters_g.npy', p, allow_pickle=True)

gmm = GMM()
gmm.save_weights()

