import numpy as np
from typing import List

class Hopfield_NN:
    def __init__(self, n, teaching_rate = 0.01, k = 0.8, stable = 7):
        self.n = n
        self.input = np.array([[0.0] for i in range(n)])
        self.weights = np.array([[0 for j in range(n)] for i in range(n)])
        self.teaching_rate = teaching_rate
        self.k = k
        self.stable = stable

    def sigmoid_f(self, arr : np.ndarray):
        """
        activation function
        """
        return 1/(1+np.exp(-arr)) -0.5

    def get_out(self, input_ : np.ndarray):
        """
        get the output to plot after stabilization (stable param)
        """
        out = input_
        x = []
        for i in range(self.stable):
            x.append(self.weights@out.T)
            out = self.sigmoid_f(self.weights@out.T)
        return out, x
    
    def set_weights(self, img_lst: np.ndarray):
        """
        direct method to train hopfield network
        """
        if len(img_lst) == 0:
            return
        img = img_lst[0][:]
        W = np.zeros([len(img[0,:])*len(img[:,0]),len(img[0,:])*len(img[:,0])],dtype=float)
        print("start")
        for n,i in enumerate(img_lst):
            x_s = i.reshape([len(i[0,:])*len(i[:,0]),1])
            W += x_s @ x_s.T
            print("x")
        W *= 1/len(img_lst)
        for i in range(len(W)):
            W[i,i] = 0
        if W.shape == self.weights.shape:
            self.weights = W
        else:
            raise ValueError("bad input images")
    
    def get_next_state(self, current):
        """
        symulate only one time step of network
        """
        out = current[:]
        out += self.sigmoid_f(0.001*self.weights@out.T)
        return out

    def teach_one(self, img):
        """
        iterative gradient descent method to teach network
        """
        img = img.reshape([len(img[0,:])*len(img[:,0])])
        dw = np.array([[img[i]*img[j] for j in range(len(img))] for i in range(len(img))])
        self.weights = self.weights + self.teaching_rate * dw