#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Commonwealth Cyber Initiative
# Author: Dr. Joao Santos
# GNU Radio version: 3.8.1.0

from distutils.version import StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import blocks
import numpy
from gnuradio import digital
from gnuradio import gr
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
from gnuradio.qtgui import Range, RangeWidget
import mmwave_beam_control
from gnuradio import qtgui

class demo_initial_access(gr.top_block, Qt.QWidget):

    def __init__(self, beam_period=0.05, center_freq=5.3e9, fft_len=64, file_suffix='demo', ia_interval=2, meas_period=1e-3, rx_gain=0.5, samp_rate=2e6, tx_gain=0.700):
        gr.top_block.__init__(self, "Commonwealth Cyber Initiative")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Commonwealth Cyber Initiative")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "demo_initial_access")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Parameters
        ##################################################
        self.beam_period = beam_period
        self.center_freq = center_freq
        self.fft_len = fft_len
        self.file_suffix = file_suffix
        self.ia_interval = ia_interval
        self.meas_period = meas_period
        self.rx_gain = rx_gain
        self.samp_rate = samp_rate
        self.tx_gain = tx_gain

        ##################################################
        # Variables
        ##################################################
        self.tx_min = tx_min = 28
        self.tx_max = tx_max = 36
        self.rx_min = rx_min = 28
        self.rx_max = rx_max = 36
        self.tx_beams = tx_beams = range(tx_min, tx_max+1)
        self.sync_word2 = sync_word2 = [0j, 0j, 0j, 0j, 0j, 0j, (-1+0j), (-1+0j), (-1+0j), (-1+0j), (1+0j), (1+0j), (-1+0j), (-1+0j), (-1+0j), (1+0j), (-1+0j), (1+0j), (1+0j), (1 +0j), (1+0j), (1+0j), (-1+0j), (-1+0j), (-1+0j), (-1+0j), (-1+0j), (1+0j), (-1+0j), (-1+0j), (1+0j), (-1+0j), 0j, (1+0j), (-1+0j), (1+0j), (1+0j), (1+0j), (-1+0j), (1+0j), (1+0j), (1+0j), (-1+0j), (1+0j), (1+0j), (1+0j), (1+0j), (-1+0j), (1+0j), (-1+0j), (-1+0j), (-1+0j), (1+0j), (-1+0j), (1+0j), (-1+0j), (-1+0j), (-1+0j), (-1+0j), 0j, 0j, 0j, 0j, 0j]
        self.sync_word1 = sync_word1 = [0., 0., 0., 0., 0., 0., 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 0., 0., 0., 0., 0.]
        self.rx_beams = rx_beams = range(rx_min, rx_max+1)
        self.pilot_symbols = pilot_symbols = ((1, 1, 1, -1,),)
        self.pilot_carriers = pilot_carriers = ((-21, -7, 7, 21,),)
        self.occupied_carriers = occupied_carriers = (list(range(-26, -21)) + list(range(-20, -7)) + list(range(-6, 0)) + list(range(1, 7)) + list(range(8, 21)) + list(range(22, 27)),)
        self.iadr = iadr = 1
        self.beam = beam = 0.010

        ##################################################
        # Blocks
        ##################################################
        self.qtgui_tab_widget_0 = Qt.QTabWidget()
        self.qtgui_tab_widget_0_widget_0 = Qt.QWidget()
        self.qtgui_tab_widget_0_layout_0 = Qt.QBoxLayout(Qt.QBoxLayout.TopToBottom, self.qtgui_tab_widget_0_widget_0)
        self.qtgui_tab_widget_0_grid_layout_0 = Qt.QGridLayout()
        self.qtgui_tab_widget_0_layout_0.addLayout(self.qtgui_tab_widget_0_grid_layout_0)
        self.qtgui_tab_widget_0.addTab(self.qtgui_tab_widget_0_widget_0, 'CCI Software-defined mmWave Platform')
        self.top_grid_layout.addWidget(self.qtgui_tab_widget_0)
        self._iadr_range = Range(0.1, 1, 0.01, 1, 200)
        self._iadr_win = RangeWidget(self._iadr_range, self.set_iadr, 'IA Interval', "counter_slider", float)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._iadr_win, 1, 0, 1, 2)
        for r in range(1, 2):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 2):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self._beam_range = Range(0.001, 0.100, 0.001, 0.010, 200)
        self._beam_win = RangeWidget(self._beam_range, self.set_beam, 'Beam Period', "counter_slider", float)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._beam_win, 0, 0, 1, 2)
        for r in range(0, 1):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 2):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self.uhd_usrp_source_0_0 = uhd.usrp_source(
            ",".join(("", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0_0.set_subdev_spec('B:0', 0)
        self.uhd_usrp_source_0_0.set_center_freq(center_freq, 0)
        self.uhd_usrp_source_0_0.set_rx_agc(False, 0)
        self.uhd_usrp_source_0_0.set_normalized_gain(rx_gain, 0)
        self.uhd_usrp_source_0_0.set_antenna('TX/RX', 0)
        self.uhd_usrp_source_0_0.set_bandwidth(0.5*samp_rate, 0)
        self.uhd_usrp_source_0_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0_0.set_time_unknown_pps(uhd.time_spec())
        self.uhd_usrp_sink_0_0 = uhd.usrp_sink(
            ",".join(("", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
            '',
        )
        self.uhd_usrp_sink_0_0.set_subdev_spec('A:0', 0)
        self.uhd_usrp_sink_0_0.set_center_freq(center_freq, 0)
        self.uhd_usrp_sink_0_0.set_normalized_gain(tx_gain, 0)
        self.uhd_usrp_sink_0_0.set_antenna('TX/RX', 0)
        self.uhd_usrp_sink_0_0.set_bandwidth(0.5*samp_rate, 0)
        self.uhd_usrp_sink_0_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0_0.set_time_unknown_pps(uhd.time_spec())
        self._tx_min_range = Range(28, 36, 1, 28, 50)
        self._tx_min_win = RangeWidget(self._tx_min_range, self.set_tx_min, 'TX Start Beam', "counter_slider", int)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._tx_min_win, 2, 0, 1, 1)
        for r in range(2, 3):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 1):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self._tx_max_range = Range(28, 36, 1, 36, 50)
        self._tx_max_win = RangeWidget(self._tx_max_range, self.set_tx_max, 'TX End Beam', "counter_slider", int)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._tx_max_win, 2, 1, 1, 1)
        for r in range(2, 3):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(1, 2):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self._rx_min_range = Range(28, 36, 1, 28, 50)
        self._rx_min_win = RangeWidget(self._rx_min_range, self.set_rx_min, 'RX Start Beam', "counter_slider", int)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._rx_min_win, 3, 0, 1, 1)
        for r in range(3, 4):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 1):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self._rx_max_range = Range(28, 36, 1, 36, 50)
        self._rx_max_win = RangeWidget(self._rx_max_range, self.set_rx_max, 'RX End Beam', "counter_slider", int)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._rx_max_win, 3, 1, 1, 1)
        for r in range(3, 4):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(1, 2):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self.qtgui_waterfall_sink_x_0 = qtgui.waterfall_sink_c(
            1024, #size
            firdes.WIN_BLACKMAN_hARRIS, #wintype
            center_freq, #fc
            samp_rate*2, #bw
            "", #name
            1 #number of inputs
        )
        self.qtgui_waterfall_sink_x_0.set_update_time(0.01)
        self.qtgui_waterfall_sink_x_0.enable_grid(False)
        self.qtgui_waterfall_sink_x_0.enable_axis_labels(True)



        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        colors = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_waterfall_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_waterfall_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_waterfall_sink_x_0.set_color_map(i, colors[i])
            self.qtgui_waterfall_sink_x_0.set_line_alpha(i, alphas[i])

        self.qtgui_waterfall_sink_x_0.set_intensity_range(-140, 10)

        self._qtgui_waterfall_sink_x_0_win = sip.wrapinstance(self.qtgui_waterfall_sink_x_0.pyqwidget(), Qt.QWidget)
        self.qtgui_tab_widget_0_grid_layout_0.addWidget(self._qtgui_waterfall_sink_x_0_win, 4, 0, 20, 2)
        for r in range(4, 24):
            self.qtgui_tab_widget_0_grid_layout_0.setRowStretch(r, 1)
        for c in range(0, 2):
            self.qtgui_tab_widget_0_grid_layout_0.setColumnStretch(c, 1)
        self.mmwave_beam_control_rss_calc_0_0 = mmwave_beam_control.rss_calc(1000, 4000)
        self.mmwave_beam_control_rate_measure_0 = mmwave_beam_control.rate_measure("/home/user/rate_meas_" + file_suffix + ".log", 1e-3, False)
        self.mmwave_beam_control_kpi_agg_0 = mmwave_beam_control.kpi_agg("/home/user/kpi_beam_" + file_suffix + ".log", "/home/user/kpi_meas_" + file_suffix + ".log", False, meas_period, -90.0, True)
        self.mmwave_beam_control_beam_sweep_0 = mmwave_beam_control.beam_sweep(False, tx_beams, rx_beams, beam, iadr, True)
        self.mmwave_beam_control_beam_selector_0 = mmwave_beam_control.beam_selector("/home/user/sel_pair_" + file_suffix + ".log", "/home/user/sel_kpi_" + file_suffix + ".log", 0.0, True)
        self.mmwave_beam_control_beam_mapper_0 = mmwave_beam_control.beam_mapper(
          tx_mhu='MHU1',
          rx_mhu='MHU2',
          backoff=2e-6,
          pulse=2e-6,
          config_path='/home/user/gr-mmwave_beam_control/examples/interdigital_mhuv3.json',
          debug=True
        )
        self.digital_ofdm_tx_0 = digital.ofdm_tx(
            fft_len=fft_len,
            cp_len=fft_len//4,
            packet_length_tag_key="packet_len",
            occupied_carriers=occupied_carriers,
            pilot_carriers=pilot_carriers,
            pilot_symbols=pilot_symbols,
            sync_word1=sync_word1,
            sync_word2=sync_word2,
            bps_header=1,
            bps_payload=1,
            rolloff=0,
            debug_log=False,
            scramble_bits=False)
        self.digital_ofdm_rx_0 = digital.ofdm_rx(
            fft_len=fft_len, cp_len=fft_len//4,
            frame_length_tag_key='frame_'+"pdu_len",
            packet_length_tag_key="pdu_len",
            occupied_carriers=occupied_carriers,
            pilot_carriers=pilot_carriers,
            pilot_symbols=pilot_symbols,
            sync_word1=sync_word1,
            sync_word2=sync_word2,
            bps_header=1,
            bps_payload=1,
            debug_log=False,
            scramble_bits=False)
        self.blocks_stream_to_tagged_stream_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, 200, "packet_len")
        self.blocks_multiply_const_vxx_0_0 = blocks.multiply_const_cc(0.1)
        self.analog_random_source_x_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 2, 1000))), True)



        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.mmwave_beam_control_beam_mapper_0, 'gpio_cmd'), (self.uhd_usrp_sink_0_0, 'command'))
        self.msg_connect((self.mmwave_beam_control_beam_selector_0, 'sweep'), (self.mmwave_beam_control_beam_sweep_0, 'sweep'))
        self.msg_connect((self.mmwave_beam_control_beam_sweep_0, 'beam_id'), (self.mmwave_beam_control_beam_mapper_0, 'beam_id'))
        self.msg_connect((self.mmwave_beam_control_beam_sweep_0, 'trigger'), (self.mmwave_beam_control_beam_selector_0, 'trigger'))
        self.msg_connect((self.mmwave_beam_control_beam_sweep_0, 'trigger'), (self.mmwave_beam_control_kpi_agg_0, 'trigger'))
        self.msg_connect((self.mmwave_beam_control_beam_sweep_0, 'beam_id'), (self.mmwave_beam_control_kpi_agg_0, 'beam_id'))
        self.msg_connect((self.mmwave_beam_control_beam_sweep_0, 'trigger'), (self.mmwave_beam_control_rate_measure_0, 'trigger'))
        self.msg_connect((self.mmwave_beam_control_kpi_agg_0, 'kpi_out'), (self.mmwave_beam_control_beam_selector_0, 'kpi_in'))
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_stream_to_tagged_stream_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0_0, 0), (self.uhd_usrp_sink_0_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0, 0), (self.digital_ofdm_tx_0, 0))
        self.connect((self.digital_ofdm_rx_0, 0), (self.mmwave_beam_control_rate_measure_0, 0))
        self.connect((self.digital_ofdm_tx_0, 0), (self.blocks_multiply_const_vxx_0_0, 0))
        self.connect((self.mmwave_beam_control_rss_calc_0_0, 0), (self.mmwave_beam_control_kpi_agg_0, 0))
        self.connect((self.uhd_usrp_source_0_0, 0), (self.digital_ofdm_rx_0, 0))
        self.connect((self.uhd_usrp_source_0_0, 0), (self.mmwave_beam_control_rss_calc_0_0, 0))
        self.connect((self.uhd_usrp_source_0_0, 0), (self.qtgui_waterfall_sink_x_0, 0))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "demo_initial_access")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_beam_period(self):
        return self.beam_period

    def set_beam_period(self, beam_period):
        self.beam_period = beam_period

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.qtgui_waterfall_sink_x_0.set_frequency_range(self.center_freq, self.samp_rate*2)
        self.uhd_usrp_sink_0_0.set_center_freq(self.center_freq, 0)
        self.uhd_usrp_source_0_0.set_center_freq(self.center_freq, 0)

    def get_fft_len(self):
        return self.fft_len

    def set_fft_len(self, fft_len):
        self.fft_len = fft_len

    def get_file_suffix(self):
        return self.file_suffix

    def set_file_suffix(self, file_suffix):
        self.file_suffix = file_suffix

    def get_ia_interval(self):
        return self.ia_interval

    def set_ia_interval(self, ia_interval):
        self.ia_interval = ia_interval

    def get_meas_period(self):
        return self.meas_period

    def set_meas_period(self, meas_period):
        self.meas_period = meas_period

    def get_rx_gain(self):
        return self.rx_gain

    def set_rx_gain(self, rx_gain):
        self.rx_gain = rx_gain
        self.uhd_usrp_source_0_0.set_normalized_gain(self.rx_gain, 0)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.qtgui_waterfall_sink_x_0.set_frequency_range(self.center_freq, self.samp_rate*2)
        self.uhd_usrp_sink_0_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_sink_0_0.set_bandwidth(0.5*self.samp_rate, 0)
        self.uhd_usrp_source_0_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0_0.set_bandwidth(0.5*self.samp_rate, 0)

    def get_tx_gain(self):
        return self.tx_gain

    def set_tx_gain(self, tx_gain):
        self.tx_gain = tx_gain
        self.uhd_usrp_sink_0_0.set_normalized_gain(self.tx_gain, 0)

    def get_tx_min(self):
        return self.tx_min

    def set_tx_min(self, tx_min):
        self.tx_min = tx_min
        self.set_tx_beams(range(self.tx_min, self.tx_max+1))

    def get_tx_max(self):
        return self.tx_max

    def set_tx_max(self, tx_max):
        self.tx_max = tx_max
        self.set_tx_beams(range(self.tx_min, self.tx_max+1))

    def get_rx_min(self):
        return self.rx_min

    def set_rx_min(self, rx_min):
        self.rx_min = rx_min
        self.set_rx_beams(range(self.rx_min, self.rx_max+1))

    def get_rx_max(self):
        return self.rx_max

    def set_rx_max(self, rx_max):
        self.rx_max = rx_max
        self.set_rx_beams(range(self.rx_min, self.rx_max+1))

    def get_tx_beams(self):
        return self.tx_beams

    def set_tx_beams(self, tx_beams):
        self.tx_beams = tx_beams
        self.mmwave_beam_control_beam_sweep_0.set_tx_iterable(self.tx_beams)

    def get_sync_word2(self):
        return self.sync_word2

    def set_sync_word2(self, sync_word2):
        self.sync_word2 = sync_word2

    def get_sync_word1(self):
        return self.sync_word1

    def set_sync_word1(self, sync_word1):
        self.sync_word1 = sync_word1

    def get_rx_beams(self):
        return self.rx_beams

    def set_rx_beams(self, rx_beams):
        self.rx_beams = rx_beams
        self.mmwave_beam_control_beam_sweep_0.set_rx_iterable(self.rx_beams)

    def get_pilot_symbols(self):
        return self.pilot_symbols

    def set_pilot_symbols(self, pilot_symbols):
        self.pilot_symbols = pilot_symbols

    def get_pilot_carriers(self):
        return self.pilot_carriers

    def set_pilot_carriers(self, pilot_carriers):
        self.pilot_carriers = pilot_carriers

    def get_occupied_carriers(self):
        return self.occupied_carriers

    def set_occupied_carriers(self, occupied_carriers):
        self.occupied_carriers = occupied_carriers

    def get_iadr(self):
        return self.iadr

    def set_iadr(self, iadr):
        self.iadr = iadr
        self.mmwave_beam_control_beam_sweep_0.set_ia_interval(self.iadr)

    def get_beam(self):
        return self.beam

    def set_beam(self, beam):
        self.beam = beam
        self.mmwave_beam_control_beam_sweep_0.set_beam_period(self.beam)


def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--beam-period", dest="beam_period", type=eng_float, default="50.0m",
        help="Set beam_period [default=%(default)r]")
    parser.add_argument(
        "--fft-len", dest="fft_len", type=intx, default=64,
        help="Set fft_len [default=%(default)r]")
    parser.add_argument(
        "--file-suffix", dest="file_suffix", type=str, default='demo',
        help="Set file_suffix [default=%(default)r]")
    parser.add_argument(
        "--ia-interval", dest="ia_interval", type=eng_float, default="2.0",
        help="Set ia_interval [default=%(default)r]")
    parser.add_argument(
        "--meas-period", dest="meas_period", type=eng_float, default="1.0m",
        help="Set meas_period [default=%(default)r]")
    parser.add_argument(
        "--rx-gain", dest="rx_gain", type=eng_float, default="500.0m",
        help="Set rx_gain [default=%(default)r]")
    parser.add_argument(
        "--tx-gain", dest="tx_gain", type=eng_float, default="700.0m",
        help="Set tx_gain [default=%(default)r]")
    return parser


def main(top_block_cls=demo_initial_access, options=None):
    if options is None:
        options = argument_parser().parse_args()

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(beam_period=options.beam_period, fft_len=options.fft_len, file_suffix=options.file_suffix, ia_interval=options.ia_interval, meas_period=options.meas_period, rx_gain=options.rx_gain, tx_gain=options.tx_gain)
    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    def quitting():
        tb.stop()
        tb.wait()
    qapp.aboutToQuit.connect(quitting)
    qapp.exec_()


if __name__ == '__main__':
    main()
