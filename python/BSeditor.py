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

#######BEAM SWEEEEEEEEEEP FILE
import logging
from gnuradio import gr
import numpy as np
import pmt
from time import sleep
from threading import Thread, Event

from datetime import datetime


class beam_sweep(gr.basic_block):
    """
    docstring for block beam_sweep
    """
    def __init__(self,
                 standalone=False,
                 tx_iterable=[32],
                 rx_iterable=[32],
                 beam_period=0.1,
                 ia_interval=5,
                 debug=False):

        gr.basic_block.__init__(
            self,
            name="Beam Sweep",
            in_sig=None,
            out_sig=None
        )

        # Check whether the iteratores are iterable
        if not hasattr(tx_iterable, '__iter__'):
            raise TypeError("Could not create the TX iterator from:",
                            tx_iterable)

        elif not hasattr(rx_iterable, '__iter__'):
            raise TypeError("Could not create the RX iterator from:",
                            rx_iterable)

        # Find out who it the outer iterable
        self._outer_iterable = tx_iterable
        self._inner_iterable = rx_iterable
        self._beam_period = beam_period
        self._ia_interval = ia_interval

        self._temp_outer_iterable = None
        self._temp_inner_iterable = None
        self._temp_beam_period = None
        self._temp_ia_interval = None

        self._tx_change_iterable = False
        self._rx_change_iterable = False
        self._change_beam_period = False
        self._change_ia_interval = False

        self.standalone = standalone

        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='[%(levelname)s] [%(name)s] %(message)s'
        )
        self.logging = logging.getLogger(self.name())

        # Create a thread to run our sweep
        self._thread = Thread(target=self.sweep)

        # Control flag to keep thread alive
        self._finished = Event()
        self.new_set_beam = None
        self._counter = 0
        self.newRx_beampattern = []
        self.newTx_beampattern = []

        # Register message port
        self.message_port_register_in(pmt.intern('sweep'))
        self.message_port_register_out(pmt.intern('beam_id'))
        self.message_port_register_out(pmt.intern('trigger'))
        # Assign sweep CTL message handler
        self.set_msg_handler(pmt.intern('sweep'), self.sweep_msg_handler)



    def pmt_publish(self, tx_index, rx_index):
        """
        Factory function to facilitate the creation of PMT messages
        """
        # Convert GPIO dict to PMT and sent control port message
        self.message_port_pub(
            pmt.intern('beam_id'),
            pmt.to_pmt({"tx": tx_index , "rx": rx_index})
        )


    def start(self):
        """
        Called at the beginning of the flowgraph execution to allocate resources
        """

        # Start our thread
        self._thread.start()

        return gr.basic_block.start(self)


    def stop(self):
        """
        Called at the end of the flowgraph execution to free resources
        """
        # Toggle flag to stop thread and join it
        self._finished.set()
        self._thread.join()

        #  self.log.close()

        return gr.basic_block.stop(self)

    def sweep(self):
        sleep(0.1)
        """
        Periodically sweeps the available beams and generate PMT messages
        """
        # While our thread is going on
        while not self._finished.is_set():
            if not self.standalone:
                # Increment counter
                self._counter += 1
                # Report state
                self.logging.info(f'Start beam_swwweeeep the IA procedure #{self._counter}')
                self.logging.info(f'this is working')
                # Let's get the party started
                self.message_port_pub(pmt.intern('trigger'), pmt.to_pmt({'trigger': True}))

                    

            # Handle changing iterables at every loop
            if self._tx_change_iterable:
                self._outer_iterable = self._temp_outer_iterable
                self._tx_change_iterable = False

            if self._rx_change_iterable:
                self._inner_iterable = self._temp_inner_iterable
                self._rx_change_iterable = False

            if self._change_beam_period:
                self._beam_period = self._temp_beam_period
                self._change_beam_period = False

            if self._change_ia_interval:
                self._ia_interval = self._temp_ia_interval
                self._change_ia_interval = False

            # Cycle through the outer loop
            for outer_index in self._outer_iterable:
                # Cycle through the inner loop
                for inner_index in self._inner_iterable:
                    # Sweep to the next beam
                    self.pmt_publish(tx_index=outer_index, rx_index=inner_index)
                    # Wait the beam period
                    self._finished.wait(self._beam_period)

            if not self.standalone:
                # Stop the sweeping
                self.message_port_pub(pmt.intern('trigger'), pmt.to_pmt({'trigger': False}))

                # Report state
                self.logging.info(f'Stop the IA procedure #{self._counter}')

                while not self.new_set_beam and not self._finished.is_set():
                    sleep(1e-9)

                if self._finished.is_set():
                    break

                # Select the best beam so far
                self.pmt_publish(
                    tx_index=self.new_set_beam['tx'],
                    rx_index=self.new_set_beam['rx']
                )
                # self.pmt_publish(
                #     tx_pattern = self.newTx_beampattern['tx'],
                #     rx_pattern = self.newRx_beampattern['rx']
                # )
                # Wait the reconfiguration time
                self._finished.wait(self._ia_interval)
                # Toggle variable back off
                self.new_set_beam = None

    ########################################################3

    def sweep_msg_handler(self, msg):
        # Convert message to python
        p_msg = pmt.to_python(msg)
        # Print debug information
        self.logging.debug(f'Received sweep message: {p_msg}')

        # Check if we receive a new start beam
        if 'set_beam' not in p_msg:
            # Raise error
            raise ValueError('Missing references to a beam: ' + str(p_msg))

        # Set the new value
        self.new_set_beam = p_msg.get('set_beam', {'tx': 0, 'rx': 0})
        self.new_beampattern = p_msg.get("set_pattern", {'TX':[], 'RX': []})

        
        if 'set_pattern' not in p_msg:
            # Raise error
            raise ValueError('Missing next TX beam pattern', {'TX': [], 'RX': []})

        # Set the new value
        self.newRx_beampattern = p_msg.get('set_pattern', {'TX': [], 'RX':[]})

        
        
        self.logging.info(f'ADRIANANANANN{(self.new_beampattern["RX"])} ADRIANANANANNADRIANANANANNADRIANANANANN:{(self.new_beampattern["TX"])}') 
        self.set_rx_iterable(self.new_beampattern['RX'])
        self.set_tx_iterable(self.new_beampattern['TX'])

    def set_tx_iterable(self, tx_iterable):
        self._temp_outer_iterable = tx_iterable
        self._tx_change_iterable = True
        self.logging.info(f'Changing TX iterable to {self._temp_outer_iterable}')

    def set_rx_iterable(self, rx_iterable):
        self._temp_inner_iterable = rx_iterable
        self._rx_change_iterable = True
        self.logging.info(f'Changing RX iterable to {self._temp_inner_iterable}')

    def set_beam_period(self, beam_period):
        self._temp_beam_period =  beam_period
        self._change_beam_period = True
        self.logging.info(f'Changing Beam Period to {self._temp_beam_period}')

    def set_ia_interval(self, ia_interval):
        self._temp_ia_interval = ia_interval
        self._change_ia_interval = True
        self.logging.info(f'Changing IA Interval to {self._temp_ia_interval}')

    #############################################################











############## END BEAM SWEEP FILE##############
        ####################################################################################################################################



import logging
import numpy as np
from gnuradio import gr
import pmt
from time import time
#from fortress import rowbotreshape1, rowtopreshape1, columnfrontreshape1, columnbackreshape1, rowbotreshape2, rowtopreshape2, columnfrontreshape2, columnbackreshape2

###############################

def rowtopreshape1(row):
    if row == 0:
        rowtop = row + 0
    elif row == 1:
        rowtop = row - 0
    elif row ==6:
        rowtop = row - 2
    elif row ==5:
        rowtop = row -1
    else:
        rowtop = row - 1
        
    return rowtop

def rowbotreshape1(row):
    if row == 0:
        rowbot = row + 3
    elif row == 1:
         rowbot = row + 3
    elif row == 6:
         rowbot = row + 1
    elif row == 5:
        rowbot = row + 2
    else:
        rowbot = row +2
    
    return rowbot

def columnfrontreshape1(column):
    if column == 8:
        columnfront = column - 2
    elif column == 7:
        columnfront = column - 1
    elif column == 1:
        columnfront = column - 1
    elif column == 0:
        columnfront = column + 0
    else:
        columnfront = column -1
        
    return columnfront

def columnbackreshape1(column):
    if column == 8:
        columnback = column + 1
    elif column == 7:
        columnback = column + 2
    elif column == 1:
        columnback = column + 2
    elif column == 0:
        columnback = column + 3
    else:
        columnback = column + 2

    return columnback

#rowtop1 = rowtopreshape1(row)
#rowbot1 = rowbotreshape1(row)

#colfront1 = columnfrontreshape1(column)
#colback1 = columnbackreshape1(column)


#25 beam wide search
def rowtopreshape2(row):
    if row == 0:
        rowtop = row + 0
    elif row == 1:
        rowtop = row - 1
    elif row ==6:
        rowtop = row - 4
    elif row ==5:
        rowtop = row -3
    else:
        rowtop = row - 2
        
    return rowtop

def rowbotreshape2(row):
    if row == 0:
        rowbot = row + 5
    elif row == 1:
         rowbot = row + 4
    elif row == 6:
         rowbot = row + 1
    elif row == 5:
        rowbot = row + 2
    else:
        rowbot = row +3
    
    return rowbot

def columnfrontreshape2(column):
    if column == 8:
        columnfront = column - 4
    elif column == 7:
        columnfront = column - 3
    elif column == 1:
        columnfront = column - 1
    elif column == 0:
        columnfront = column + 0
    else:
        columnfront = column -2
        
    return columnfront

def columnbackreshape2(column):
    if column == 8:
        columnback = column + 1
    elif column == 7:
        columnback = column + 2
    elif column == 1:
        columnback = column + 4
    elif column == 0:
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

        # Check if we receive a new sweep state
        if 'trigger' not in p_msg:
            # Raise error
            raise ValueError('Missing references to a trigger: ' + str(p_msg))

        # Set the new valuep
        self.trigger = p_msg.get('trigger', True)


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


                if self._sel_counter >=1:
                    txnrow, txncol = self.pattern.shape
                    txMax = tx_beam
                    TXrid = txMax//txncol 
                    TXcid = txMax%txncol
                    (TXrid, TXcid)
                    TXrow = TXrid
                    TXcolumn = TXrid
                    TXrowtop = rowtopreshape1(TXrow)
                    TXrowbot = rowbotreshape1(TXrow)
                    TXcolfront = columnfrontreshape1(TXcolumn)
                    TXcolback = columnbackreshape1(TXcolumn)
                    tx = self.pattern[TXrowtop:TXrowbot, TXcolfront:TXcolback]
                    #tx = btx.flatten

                    rxnrow, rxncol = self.pattern.shape
                    RxMax = rx_beam
                    RXrid = RxMax//rxncol 
                    RXcid = RxMax%rxncol
                    (RXrid, RXcid)
                    RXrow = RXrid
                    RXcolumn = RXrid
                    RXrowtop = rowtopreshape1(RXrow)
                    RXrowbot = rowbotreshape1(RXrow)
                    RXcolfront = columnfrontreshape1(RXcolumn)
                    RXcolback = columnbackreshape1(RXcolumn)
                    rx = self.pattern[RXrowtop:RXrowbot, RXcolfront:RXcolback]
                    #rx = brx.flatten()
                
                else:
                    tx = self.pattern
                    #tx = btx.flatten()
                    rx = self.pattern
                    #rx = brx.flatten()


            # Measure elapsed time
            elapsed = time() - start

            if self._beam_store:
                # Report findings
                self.logging.info(f'ADRIAN Pair TX {tx_beam} RX {rx_beam} RSS {kpi} Time {elapsed}')
                self.logging.info(f'Pattern TX {tx} RX {rx}')
                self.logging.info(f'ADRIANSADRIANSADRIANSbeampair RX {rx} TX {tx}')

                # Use the best beam
                self.message_port_pub(
                    pmt.intern('sweep'),
                    pmt.to_pmt({"set_beam":{'tx': tx_beam, 'rx': rx_beam}, "set_pattern": {'TX':(tx.flatten()).tolist(), 'RX': (rx.flatten()).tolist()}})
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
