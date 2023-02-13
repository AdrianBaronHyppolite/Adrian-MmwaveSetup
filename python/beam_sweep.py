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
from gnuradio import gr
import numpy as np
import pmt
from time import sleep
from time import time
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
            start = time()
            if not self.standalone:
                # Increment counter
                self._counter += 1
                # Report state
                self.logging.info(f'Start beam_sweep the IA procedure #{self._counter}')
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

            elapsed = time() -start
            self.logging.info(f'sweep time:{elapsed}')


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

        
        
        self.logging.info(f'RX beam pattern{(self.new_beampattern["RX"])} TX beam pattern:{(self.new_beampattern["TX"])}') 
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