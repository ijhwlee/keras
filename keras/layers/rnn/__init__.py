# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras recurrent layers."""

import tensorflow as tf

from tf.python.keras.layers.rnn.abstract_rnn_cell import AbstractRNNCell

# Recurrent layers.
from tf.python.keras.layers.rnn.base_rnn import RNN
from tf.python.keras.layers.rnn.simple_rnn import SimpleRNN
from tf.python.keras.layers.rnn.simple_rnn import SimpleRNNCell
from tf.python.keras.layers.rnn.stacked_rnn_cells import StackedRNNCells

if tf.__internal__.tf2.enabled():
    from tf.python.keras.layers.rnn.gru import GRU
    from tf.python.keras.layers.rnn.gru import GRUCell
    from tf.python.keras.layers.rnn.gru_v1 import GRU as GRUV1
    from tf.python.keras.layers.rnn.gru_v1 import GRUCell as GRUCellV1
    from tf.python.keras.layers.rnn.lstm import LSTM
    from tf.python.keras.layers.rnn.lstm import LSTMCell
    from tf.python.keras.layers.rnn.lstm_v1 import LSTM as LSTMV1
    from tf.python.keras.layers.rnn.lstm_v1 import LSTMCell as LSTMCellV1

    GRUV2 = GRU
    GRUCellV2 = GRUCell
    LSTMV2 = LSTM
    LSTMCellV2 = LSTMCell
else:
    from tf.python.keras.layers.rnn.gru import GRU as GRUV2
    from tf.python.keras.layers.rnn.gru import GRUCell as GRUCellV2
    from tf.python.keras.layers.rnn.gru_v1 import GRU
    from tf.python.keras.layers.rnn.gru_v1 import GRUCell
    from tf.python.keras.layers.rnn.lstm import LSTM as LSTMV2
    from tf.python.keras.layers.rnn.lstm import LSTMCell as LSTMCellV2
    from tf.python.keras.layers.rnn.lstm_v1 import LSTM
    from tf.python.keras.layers.rnn.lstm_v1 import LSTMCell

    GRUV1 = GRU
    GRUCellV1 = GRUCell
    LSTMV1 = LSTM
    LSTMCellV1 = LSTMCell

# Wrapper functions.
from tf.python.keras.layers.rnn.base_wrapper import Wrapper
from tf.python.keras.layers.rnn.bidirectional import Bidirectional

# RNN Cell wrappers.
from tf.python.keras.layers.rnn.cell_wrappers import DeviceWrapper
from tf.python.keras.layers.rnn.cell_wrappers import DropoutWrapper
from tf.python.keras.layers.rnn.cell_wrappers import ResidualWrapper

# Convolutional-recurrent layers.
from tf.python.keras.layers.rnn.conv_lstm1d import ConvLSTM1D
from tf.python.keras.layers.rnn.conv_lstm2d import ConvLSTM2D
from tf.python.keras.layers.rnn.conv_lstm3d import ConvLSTM3D
from tf.python.keras.layers.rnn.cudnn_gru import CuDNNGRU

# cuDNN recurrent layers.
from tf.python.keras.layers.rnn.cudnn_lstm import CuDNNLSTM
from tf.python.keras.layers.rnn.time_distributed import TimeDistributed
