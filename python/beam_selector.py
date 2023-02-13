#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 Virginia Tech.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#
import logging
import numpy as np
import torch
from gnuradio import gr
import pmt
from gym import Env
from time import time
from gym.spaces import Discrete, Box
#from fortress import rowbotreshape1, rowtopreshape1, columnfrontreshape1, columnbackreshape1, rowbotreshape2, rowtopreshape2, columnfrontreshape2, columnbackreshape2

############################
#multi armed bandit
class eps_bandit:
    '''
    epsilon-greedy k-bandit problem
    
    Inputs
    =====================================================
    k: number of arms (int)
    eps: probability of random action 0 < eps < 1 (float)
    iters: number of steps (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a normal distribution with mean = 0. 
        Set to "sequence" for the means to be ordered from 
        0 to k-1.
        Pass a list or array of length = k for user-defined
        values.
    '''
    
    def __init__(self, k, eps, iters, mu='random'):
        # Number of arms
        self.k = k
        # Search probability
        self.eps = eps
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 0
        # Step count for each arm
        self.k_n = np.zeros(k)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(iters)
        # Mean reward for each arm
        self.k_reward = np.zeros(k)
        
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)

        np.random.seed(13)
        
    def act(self, eps = 0):
        p = np.random.rand()
        if p < eps:
            # Randomly select an action
            a = np.random.choice(self.k)
        else:
            # Take greedy action
            a = np.argmax(self.k_reward)
        
        return a
    
    def learn(self, reward, action):
        # Update counts
        self.n += 1
        self.k_n[action] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[action] = self.k_reward[action] + (
            reward - self.k_reward[action]) / self.k_n[action]
            
    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(self.iters)
        self.k_reward = np.zeros(self.k)
############################


##############################
#contextual multiarmed bandit
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Deep Q-network.
"""
class DQN(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super(DQN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_actions))
        return 

    def forward(self, x):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x, device=DEVICE)
        y = self.net(x.float())
        return y

class contextual_bandit(eps_bandit):
    '''
    contextual epsilon-greedy k-bandit problem
    
    Inputs
    =====================================================
    k: number of arms (int)
    eps: probability of random action 0 < eps < 1 (float)
    iters: number of steps (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a normal distribution with mean = 0. 
        Set to "sequence" for the means to be ordered from 
        0 to k-1.
        Pass a list or array of length = k for user-defined
        values.
    '''
    
    def __init__(self, k, eps, iters, hidden_size, state_size, mu='random', learning_rate=0.5, qnet=None):
        # Number of arms
        self.k = k
        # Search probability
        self.eps = eps
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 0
        # Step count for each arm
        self.k_n = np.zeros(k)
        
        # Total mean reward
        self.mean_reward = 0
        
        # Mean reward for each arm
        self.reward = np.zeros(iters)
        
        self.hidden_size=hidden_size
        self.state_size=state_size
        
        self.neuralNet = DQN(state_dim=self.state_size, hidden_dim=self.hidden_size, num_actions=self.k).to(DEVICE)
        self.opt = torch.optim.Adam(params=self.neuralNet.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)

        np.random.seed(13)
        torch.manual_seed(13)
        
    def act(self, state, eps = 0):
        with torch.no_grad():
            self.neuralNet.eval()
            q_values = self.neuralNet(state)
        # Random action.
        if np.random.rand() < eps:
            nA = q_values.shape[-1]
            action = np.random.choice(nA)
        # Greedy policy.
        else:
            action = q_values.argmax().cpu().numpy()
        return action
    
    def learn(self, reward, state, action):
        # Update counts
        self.n += 1
        self.k_n[action] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n

        self.neuralNet.train()
        q_values = self.neuralNet(state)
        y = q_values.max()

        # Update results for a_k
        with torch.no_grad():
            yhat = y + (reward - y) / self.k_n[action]
            yhat = yhat.reshape(y.shape).float()
        
        # # Back-propagation.
        self.opt.zero_grad()
        loss = self.loss_fn(y, yhat)
        loss.backward()
        self.opt.step()
        
    def incrementIters(self, increment): 
        self.iters += int(increment)
        return

    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(self.iters)
        self.neuralNet = np.zeros(self.k)
        return
##############################


###############################

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
        strattx = 0
        stratrx = 0
        return rx, tx, strattx, stratrx
    

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
        rx = codebook[row, column]
        tx = codebook[txrow, txcolumn]
        strattx = 1
        stratrx = 1
        return rx, tx, strattx, stratrx
    
    
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
        strattx = 2
        stratrx = 2
        return rx, tx, strattx, stratrx
    

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
        strattx = 3
        stratrx = 2
        return rx, tx, strattx, stratrx

    # reward function calculation
    def get_rxmatrix(self):
        return self.state[0]
    
    def get_txmatrix(self):
        return self.state[1]

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
#################################

#localsearch1
def rowtopreshape1(row, a):
    zerorow = a.shape[0]-a.shape[0]
    onerow = a.shape[0]- (a.shape[0]-1)
    lastrow = a.shape[0]-1
    secondtolastrow = a.shape[0]-2


    if row == zerorow:
        rowtop = row + 0
    elif row == onerow:
        rowtop = row - 0
    elif row ==lastrow:
        rowtop = row - 2
    elif row ==secondtolastrow:
        rowtop = row -1
    else:
        rowtop = row - 1
        
    return rowtop

def rowbotreshape1(row, a):
    zerorow = a.shape[0]-a.shape[0]
    onerow = a.shape[0]- (a.shape[0]-1)
    lastrow = a.shape[0]-1
    secondtolastrow = a.shape[0]-2

    if row == zerorow:
        rowbot = row + 3
    elif row == onerow:
         rowbot = row + 3
    elif row == lastrow:
         rowbot = row + 1
    elif row == secondtolastrow:
        rowbot = row + 2
    else:
        rowbot = row +2
    
    return rowbot

def columnfrontreshape1(column, a):
    zerocolumn = a.shape[1]-a.shape[1]
    onecolumn = a.shape[1]- (a.shape[1]-1)
    lastcolumn = a.shape[1]-1
    secondtolastcolumn = a.shape[1]-2

    if column == lastcolumn:
        columnfront = column - 2
    elif column == secondtolastcolumn:
        columnfront = column - 1
    elif column == onecolumn:
        columnfront = column - 1
    elif column == zerocolumn:
        columnfront = column + 0
    else:
        columnfront = column -1
        
    return columnfront

def columnbackreshape1(column, a):
    zerocolumn = a.shape[1]-a.shape[1]
    onecolumn = a.shape[1]- (a.shape[1]-1)
    lastcolumn = a.shape[1]-1
    secondtolastcolumn = a.shape[1]-2

    if column == lastcolumn:
        columnback = column + 1
    elif column == secondtolastcolumn:
        columnback = column + 2
    elif column == onecolumn:
        columnback = column + 2
    elif column == zerocolumn:
        columnback = column + 3
    else:
        columnback = column + 2

    return columnback

#25 beam wide search
def rowtopreshape2(row, a):
    zerorow = a.shape[0]-a.shape[0]
    onerow = a.shape[0]- (a.shape[0]-1)
    lastrow = a.shape[0]-1
    secondtolastrow = a.shape[0]-2


    if row == zerorow:
        rowtop = row + 0
    elif row == onerow:
        rowtop = row - 1
    elif row ==lastrow:
        rowtop = row - 4
    elif row ==secondtolastrow:
        rowtop = row -3
    else:
        rowtop = row - 2
        
    return rowtop

def rowbotreshape2(row, a):
    zerorow = a.shape[0]-a.shape[0]
    onerow = a.shape[0]- (a.shape[0]-1)
    lastrow = a.shape[0]-1
    secondtolastrow = a.shape[0]-2

    if row == zerorow:
        rowbot = row + 5
    elif row == onerow:
         rowbot = row + 4
    elif row == lastrow:
         rowbot = row + 1
    elif row == secondtolastrow:
        rowbot = row + 2
    else:
        rowbot = row +3
    
    return rowbot

def columnfrontreshape2(column, a):
    zerocolumn = a.shape[1]-a.shape[1]
    onecolumn = a.shape[1]- (a.shape[1]-1)
    lastcolumn = a.shape[1]-1
    secondtolastcolumn = a.shape[1]-2

    if column == lastcolumn:
        columnfront = column - 4
    elif column == secondtolastcolumn:
        columnfront = column - 3
    elif column == onecolumn:
        columnfront = column - 1
    elif column == zerocolumn:
        columnfront = column + 0
    else:
        columnfront = column -2
        
    return columnfront

def columnbackreshape2(column, a):
    zerocolumn = a.shape[1]-a.shape[1]
    onecolumn = a.shape[1]- (a.shape[1]-1)
    lastcolumn = a.shape[1]-1
    secondtolastcolumn = a.shape[1]-2

    if column == lastcolumn:
        columnback = column + 1
    elif column == secondtolastcolumn:
        columnback = column + 2
    elif column == onecolumn:
        columnback = column + 4
    elif column == zerocolumn:
        columnback = column + 5
    else:
        columnback = column + 3
    return columnback

###########################################3

class beam_selector(gr.basic_block):
    """
    Method that selects the best beam for IA
    """
    def __init__(self,
             pair_file="/home/adrian/sel_pair.log",
             kpi_file="/home/adrian/sel_kpi.log",
             threshold=0.0,
             debug=False,
        ):

        #self.logging.info(f'LOGLOGLOGLOGLOGLOGLOGLOGLOGLOG\n\n\n\n\n')

        self.pattern = np.arange(63)
        self.pattern.shape = (7,9)
        # Check if the threshold is not a positiver number
        if threshold < 0.0:
            raise ValueError("Negative threshold:" + str(threshold) )

        gr.basic_block.__init__(self,
            name="Beam Selector",
            in_sig=None,
            out_sig=None)

        # Save parameters as class variables
        self._threshold = threshold

        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='[%(levelname)s] [%(name)s] %(message)s'
        )
        self.logging = logging.getLogger(self.name())

        self._sel_counter = 0
        self._kpi_counter = 0
        self._beam_store = {}

        self.env = None

        # Register message port
        self.message_port_register_in(pmt.intern('trigger'))
        self.message_port_register_in(pmt.intern('kpi_in'))
        self.message_port_register_out(pmt.intern('sweep'))


        # Assign sweep CTL message handler
        self.set_msg_handler(pmt.intern('trigger'), self.trigger_msg_handler)
        self.set_msg_handler(pmt.intern('kpi_in'), self.val_msg_handler)

        # Open file and add headers
        self.results = open(pair_file, "w")
        self.results.write("#,TX,RX,KPI,elapsed\n")

        self.kpi = open(kpi_file, "w")
        self.kpi.write("#,TX,RX,KPI\n")

    def stop(self):
        """
        Called at the end of the flowgraph execution to free resources
        """
        self.results.close()
        self.kpi .close()

        return gr.basic_block.stop(self)


    def val_msg_handler(self, msg):
        # Convert message to python
        p_msg = pmt.to_python(msg)
        # Print debug information
        self.logging.debug(f'Received Value message: {p_msg}')

        # Check if we receive a new sweep state
        kpi = p_msg.get('val', -999.0)
        # Check if we receive  a beam ID for the TX
        tx_beam = p_msg.get('tx', 32)
        # Check if we receive  a beam ID for the RX
        rx_beam = p_msg.get('rx', 32)

        # Check if we need to create a new entry in the beam store
        if (tx_beam, rx_beam) not in self._beam_store:
            self._beam_store[(tx_beam, rx_beam)] = []

        if self.trigger:
            self._beam_store[(tx_beam, rx_beam)].append(kpi)

        self._kpi_counter += 1
        self.kpi.write(f"{self._kpi_counter},{tx_beam},{rx_beam},{kpi}\n")

    def trigger_msg_handler(self, msg):
        # Convert message to python
        p_msg = pmt.to_python(msg)
        # Print debug information
        self.logging.debug(f'Received trigger message: {p_msg}')

        
        #Call an instance of the RLIA class
        # Training setup Contextual MAB    
        eps=0.1
        lr= 0.15
        training_episodes = 1000
        contextual_agent = contextual_bandit(k=4, eps=eps, iters=training_episodes, hidden_size=32, 
        state_size=3, learning_rate=lr)  
        mab_agent = eps_bandit(k=4, eps=eps, iters=training_episodes) 


        premeasuretx = 0
        premeasurerx = 0
        listofkpis = 0


        # Check if we receive a new sweep state
        if 'trigger' not in p_msg:
            # Raise error
            raise ValueError('Missing references to a trigger: ' + str(p_msg))

        # Set the new valuep
        self.trigger = p_msg.get('trigger', True)

        #initialize environment.

        # When triggered, reset saved information
        if self.trigger:
            self._sel_counter += 1
            self._beam_store = {}

        else:
            start = time()
            # Get the average KPI values per beam pair
            for beam_pair in self._beam_store:
                self._beam_store[beam_pair] = np.median(self._beam_store[beam_pair])

            if self._beam_store:
                # Extract the beam pair with highest KPI
                tx_beam, rx_beam = max(self._beam_store, key=self._beam_store.get)
                kpi = self._beam_store[(tx_beam, rx_beam)]
                # initialize premeasures to determine first ideal beam pair
            if self._sel_counter == 1:
                premeasurerx = rx_beam
                premeasuretx = tx_beam
                self.env = RLIAEnv(premeasuretx, premeasurerx)

            if self._sel_counter >=1:
                for _ in range(self._sel_counter):
                    action = mab_agent.act(eps)
                    _ , r, _ , _ = self.env.step(action, kpi)

                    # Collect reward.
                    mab_agent.learn(reward=r, action=action) 

                    #Training contextual MAB
                    #flat_state = np.array([self.env.state[2], self.env.state[3], rx_beam])
                    #action = contextual_agent.act(flat_state)
                    #_ , r , _ , _ = self.env.step(action = action)
                    #contextual_agent.learn(reward=r, state=flat_state, action=action)
                    pack_txrx = self.env.act(action)
                    tx = pack_txrx[0]
                    rx = pack_txrx[1]
            else:
                tx = self.pattern
                rx = self.pattern
                   
                    


            # Measure elapsed time
            elapsed = time() - start

            if self._beam_store:
                # Report findings
                self.logging.info(f'ADRIAN Pair TX {tx_beam} RX {rx_beam} RSS {kpi} Time {elapsed}')
                self.logging.info(f'Pattern TX {tx} RX {rx}')
                self.logging.info(f'ADRIAN beampair RX {rx} TX {tx}')
                self.logging.info(f'Action choice{kpi}')

                # Use the best beam
                self.message_port_pub(
                    pmt.intern('sweep'),
                    pmt.to_pmt({"set_beam":{'tx': tx_beam, 'rx': rx_beam}, "set_pattern": {'TX':tx.flatten().tolist(), 'RX': rx.flatten().tolist()}})
                    #.flatten()).tolist())
                )
                self.results.write(f"{self._sel_counter},{tx_beam},{rx_beam},{tx},{rx},{kpi},{elapsed}\n")

            else:
                # Report findings
                self.logging.info(f'Failed IA, could find find a beam pair')

                # Use the best beam
                self.message_port_pub(
                    pmt.intern('sweep'),
                    pmt.to_pmt({"set_beam": {'tx': 0, 'rx': 0}, "set_pattern": {'TX':[], 'RX': []}})
                )
                self.results.write(f"{self._sel_counter},0,0,0,{elapsed}\n")
