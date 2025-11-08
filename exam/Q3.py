# If you need to import additional packages or classes, please import here.
import numpy as np
import sys
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#numpy only implementation of LSTM
class LstmParam:
    def __init__(self, hidden_size, x_dim):
        self.hidden_size = hidden_size
        self.x_dim = x_dim
        
        # Fix seed=42 for reproducibility
        # total dim = x_dim + hidden_size
        # wg, wi, wf, wo has shape (hidden_size, x_dim + hidden_size) and Uniform(-0.1, 0.1) init
        param_dim = x_dim + hidden_size
        rng = np.random.default_rng(42)
        self.wg = rng.uniform(-0.1, 0.1, (hidden_size, param_dim))
        self.wi = rng.uniform(-0.1, 0.1, (hidden_size, param_dim))
        self.wf = rng.uniform(-0.1, 0.1, (hidden_size, param_dim))
        self.wo = rng.uniform(-0.1, 0.1, (hidden_size, param_dim))
        
        # bg, bi, bf, bo has shape (hidden_size,) and init to Uniform(-0.1, 0.1)
        self.bg = rng.uniform(-0.1, 0.1, (hidden_size,))
        self.bi = rng.uniform(-0.1, 0.1, (hidden_size,))
        self.bf = rng.uniform(-0.1, 0.1, (hidden_size,))
        self.bo = rng.uniform(-0.1, 0.1, (hidden_size,))
        
class LstmState:
    def __init__(self, hidden_size, x_dim):
        # init g, i, f, o, s, h as zero shape (hidden_size,)
        self.g = np.zeros((hidden_size,))
        self.i = np.zeros((hidden_size,))
        self.f = np.zeros((hidden_size,))
        self.o = np.zeros((hidden_size,))
        self.s = np.zeros((hidden_size,))
        self.h = np.zeros((hidden_size,))
        
class LstmNode:
    def __init__(self, lstm_param, lstm_state):
        # store reference (self.param, self.state) and prepaere self.xc (xc is x concat h)
        self.param = lstm_param
        self.state = lstm_state
        self.xc = None  # x concat h will be stored here during forward
    
    def forward(self, x, s_prev, h_prev):
        # concatenate x and h_prev to xc -> shape (x_dim + hidden_size,)
        self.xc = np.hstack((x, h_prev))
        # g = tanh(Wg @ xc + bg)
        self.state.g = np.tanh(np.dot(self.param.wg, self.xc) + self.param.bg)
        # i = sigmoid(Wi @ xc + bi)
        self.state.i = sigmoid(np.dot(self.param.wi, self.xc) + self.param.bi)
        # f = sigmoid(Wf @ xc + bf)
        self.state.f = sigmoid(np.dot(self.param.wf, self.xc) + self.param.bf)
        # o = sigmoid(Wo @ xc + bo)
        self.state.o = sigmoid(np.dot(self.param.wo, self.xc) + self.param.bo)
        # s = g * i + s_prev * f
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        # h = tanh(s) * o
        self.state.h = np.tanh(self.state.s) * self.state.o
        # return (s, h)
        return self.state.s, self.state.h
    
class LstmNetwork():
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_node_list = []
        self.x_list = []
        
    def x_list_clear(self):
        self.x_list = []
        self.lstm_node_list = []
        
    def x_list_add(self, x):
        # append x to self.x_list
        self.x_list.append(x)
        # get (s_prev, h_prev) = zero if first x, else from last lstm_node in self.lstm_node_list
        if len(self.lstm_node_list) == 0:
            s_prev = np.zeros((self.lstm_param.hidden_size,))
            h_prev = np.zeros((self.lstm_param.hidden_size,))
        else:
            last_node = self.lstm_node_list[-1]
            s_prev = last_node.state.s
            h_prev = last_node.state.h
        # create new LstmState and LstmNode, call forward with (x, s_prev, h_prev), append node
        lstm_state = LstmState(self.lstm_param.hidden_size, self.lstm_param.x_dim)
        lstm_node = LstmNode(self.lstm_param, lstm_state)
        lstm_node.forward(x, s_prev, h_prev)
        self.lstm_node_list.append(lstm_node)

def func():
    # Driver: do not modify
    np.random.seed(42)
    sequence_length = 5
    x_dim = 7
    hidden_size = 5

    lstm_param = LstmParam(hidden_size, x_dim)
    lstm_net = LstmNetwork(lstm_param)

    # Two random sequences of shape (5,7)
    inputs = [
        np.random.rand(sequence_length, x_dim),
        np.random.rand(sequence_length, x_dim)
    ]

    for sample in inputs:
        for xt in sample:
            lstm_net.x_list_add(xt)
        for node in lstm_net.lstm_node_list:
            print(round(float(node.state.h[0]), 6))
        print("\n")
        lstm_net.x_list_clear()

if __name__ == "__main__":
    func()