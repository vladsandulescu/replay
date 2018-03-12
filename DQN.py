import tensorflow as tf
import numpy as np

class DQN:
    '''
    DQN with experience replay
    '''
    def __init__(self, N=1e3):
        self.N = N
        self.replay_buffer = []
