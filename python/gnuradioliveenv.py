from operator import truediv
from gym import Env
from fortress import rowbotreshape1, rowtopreshape1, columnfrontreshape1, columnbackreshape1, rowbotreshape2, rowtopreshape2, columnfrontreshape2, columnbackreshape2
from gym.spaces import Discrete, Box
import pandas as pd
import fortress
import math
import numpy as np
import random
import gym
import logging
from gnuradio import gr
import pmt
from time import time
# import pack2dict



class RLIAEnv(Env):
    def __init__(self, tx, rx):

        # The RL algorithm should select one of four beam training options
        self.action_space = Discrete(4)
        # observation space
        self.observation_space = Box(low=-50, high=-7, shape=(7,9), dtype=np.float16)
        # Environment values
        self.state = 0
        self.alpha = 0.5
        self.pp = [0.5, 0.75, 1.0, 1.5]
        self.a = -90
        self.sig = self.snr/(self.r-self.a)
        self.A_dB = -30
        self.premeasurerx = tx
        self.premeasuretx = rx
        self.pattern = np.arange(63)
        self.pattern.shape = (7,9)
        ##########
        ##########newspec = 0

    
    #exhaustive search mechanism
    def exhaustive(self, matrix):
        rx = matrix
        tx = matrix
        return rx, tx
    

    #last beam selection search mechanism
    def lastBeam(self, codebook, ptxi=1):
        nrow, ncol = codebook.shape
        i = self.premeasurerx
        rid = i//ncol
        cid = i % ncol
        (rid, cid)
        row = rid
        column = rid

        txi = self.premeasuretx
        txrid = txi//ncol
        txcid = txi % ncol
        (txrid, txcid)
        txrow = txrid
        txcolumn = txrid
        # subset of base station beams
        rx = codebook[row:row+1, column:column+1]
        tx = codebook[txrow:txrow+1, txcolumn:txcolumn+1]
        return rx, tx
    
    
    #25 beam wide search mechanism
    def localsearch1(self, codebook, ptxi=1):
        nrow, ncol = codebook.shape
        i = self.premeasurerx
        rid = i//ncol
        cid = i % ncol
        (rid, cid)
        row = rid
        column = rid
        rowtop = rowtopreshape1(row)
        rowbot = rowbotreshape1(row)
        colfront = columnfrontreshape1(column)
        colback = columnbackreshape1(column)

        txi = self.premeasuretx
        txrid = txi//ncol
        txcid = txi % ncol
        (txrid, txcid)
        txrow = txrid
        txcolumn = txrid
        txrowtop = rowtopreshape1(txrow)
        txrowbot = rowbotreshape1(txrow)
        txcolfront = columnfrontreshape1(txcolumn)
        txcolback = columnbackreshape1(txcolumn)
        # basestation beam pattern
        tx = codebook[txrowtop:txrowbot, txcolfront:txcolback]
        rx = codebook[rowtop:rowbot, colfront:colback]
        
        return rx, tx
    

    #9X9 square search mechanism
    def localsearch2(self, codebook, ptxi=1):
        nrow, ncol = codebook.shape
        i = self.premeasurerx
        rid = i//ncol
        cid = i % ncol
        (rid, cid)
        row = rid
        column = rid
        rowtop = rowtopreshape1(row)
        rowbot = rowbotreshape1(row)
        colfront = columnfrontreshape1(column)
        colback = columnbackreshape1(column)

        txi = self.premeasuretx
        txrid = txi//ncol
        txcid = txi % ncol
        (txrid, txcid)
        txrow = txrid
        txcolumn = txrid
        txrowtop = rowtopreshape2(txrow)
        txrowbot = rowbotreshape2(txrow)
        txcolfront = columnfrontreshape2(txcolumn)
        txcolback = columnbackreshape2(txcolumn)
        # basestation beam pattern
        tx = codebook[txrowtop:txrowbot, txcolfront:txcolback]
        rx = codebook[rowtop:rowbot, colfront:colback]
        return rx, tx

    # reward function calculation

    def rewardfunction(self, action, spec):
        rfunc = self.alpha * spec - (1-self.alpha) * self.pp[action]
        return rfunc

    def act(self, action_index):
        if (action_index == 0):
            set = self.lastBeam(self.pattern)
        elif (action_index == 1):
            set = self.localsearch1(self.pattern)
        elif (action_index == 2):
            set = self.localsearch2(self.pattern)
        elif (action_index == 3):
            set = self.exhaustive(self.pattern)
        return set

    def step(self, action, spec, move=False):

        self.state = self.act(action)
        self.rxmatrix = self.state[0]
        self.txmatrix = self.state[1]

        reward = self.rewardfunction(action, spec)

        if(move):
            self.move()

        return self.state, reward, True, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = 0
        self.spec = 0
        return self.state