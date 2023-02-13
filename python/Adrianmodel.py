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

import pandas as pd
import numpy as np
import tensorflow as tf
from gnuradio import gr
import pmt
import sklearn
import pickle

class classify_csi(gr.sync_block):
    """
    docstring for block classify_csi
    """
    def __init__(self,
                 model_path='~/home/adrian/RLEXPMOD',
                 weights_path='~/Downloads/gesture_classifier_weights.h5',
                 class_pickle_path='~/Downloads/label_encoder.pkl',
                 scale_pickle_path='~/Downloads/scale_encoder.pkl',
                 start_carrier=1,
                 end_carrier=63,
                 fft_size=64,
                 debug = False):

        # Sanitize input
        if fft_size < 2:
            raise ValueError('Number of subcarriers too small:', fft_size)

        # Set class variables
        self.N = fft_size
        self._start_carrier = start_carrier
        self._end_carrier = end_carrier

        # Let's try to open the JSON model file
        try:
            with open(model_path, 'rb') as model_file:
                loaded_model_file = model_file.read()
                  # If no go, raise error
        except IOError:
            raise FileNotFoundError(
                'Could not find model file: ' + str(model_path)
            )

        # Load model
        self.loaded_model = tf.keras.models.model_from_json(loaded_model_file)

        # Let's try to open the H5 weights file
        try:
            # Load weights into new model
            self.loaded_model.load_weights(weights_path)

        # If no go, raise error
        except Exception:
            raise FileNotFoundError(
                'Could not find or load weights file: ' + str(weights_path)
            )

        # Let's try to open the pickle file
        try:
            class_pickle_file = open(class_pickle_path, 'rb')

        # If no go, raise error
        except IOError:
            raise FileNotFoundError(
                'Could not find class pickle file: ' + str(class_pickle_path)
            )

        # Open Pickle file
        self.label_encoder = pickle.load(class_pickle_file)

        # Let's try to open the pickle file
        try:
            scale_pickle_file = open(scale_pickle_path, 'rb')

        # If no go, raise error
        except IOError:
            raise FileNotFoundError(
                'Could not find scale pickle file: ' + str(scale_pickle_path)
            )

        # Open Pickle file
        self.scale_list = pickle.load(scale_pickle_file)

        # Save parameters as class variables
        self._debug = debug
        self._classification = None

        gr.sync_block.__init__(self,name='Classify CSI',
                               in_sig=[(np.complex64,self.N)],
                               out_sig=None)


    def get_classification(self):
        # Return last classification
        return self._classification


    def work(self, input_items, output_items):
        # Get all the tags in the input items
        tags = self.get_tags_in_window(0, 0, len(input_items[0]))

        # Get the CSI values
        csi_values = [pmt.to_python(tag.value) for tag in tags if
                      pmt.to_python(tag.key) == 'ofdm_sync_chan_taps']

        # If we found any of them
        if csi_values:
            # Store amplitude values
            df = np.concatenate(
                [np.abs(csi_values[0]), np.angle(csi_values[0])]
            )

            # Reshape and transpose the measurements
            df = np.transpose(df.reshape(128,1)[
                self._start_carrier:self._end_carrier]
            )


            df = self.scale_list.transform(df)

            df = df.reshape(df.shape[0], 1, df.shape[1])

            #  print(df)

            # Classify CSI values
            predicted_category = np.argmax(
                self.loaded_model.predict(df), axis=-1
            )

            # Obtain a human-readable class
            self._classification = self.label_encoder.inverse_transform(
                predicted_category
            )

            # If set to debug
            if self._debug:
                # Output classification result
                print(self.name(), self._classification)

        return len(input_items[0])
