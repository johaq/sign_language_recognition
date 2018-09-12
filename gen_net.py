import datetime
from keras.layers import Dense, LSTM, RepeatVector, Flatten, Conv1D, MaxPooling1D, Concatenate, Add
import keras as K
import numpy as np


class NetGen:

    def __init__(self):
        pass

    def get_std_net(self) -> K.Sequential:
        encoder_input = K.layers.Input(shape=(
                None, 390))

        encoder = LSTM(1024, return_state=True)
        encoder_output, state_h, state_c = encoder(encoder_input)
        encoder_states = [state_h, state_c]

        decoder_input = K.layers.Input(shape=(None, 118))
        decoder_lstm = LSTM(1024, return_sequences=True, return_state=True)
        decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
        decoder_dense = K.layers.Dense(118, activation='softmax')
        decoder_output = decoder_dense(decoder_output)

        model = K.models.Model([encoder_input, decoder_input], decoder_output)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        name = self.generate_netname()
        return model, name

    def generate_netname(self):
        name = ('net_' + str(datetime.datetime.now())).replace(' ', '_')
        return name
