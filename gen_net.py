import datetime
from keras.layers import Dense, LSTM, RepeatVector, Flatten, Conv1D, MaxPooling1D, Concatenate, Add
import keras as K
import numpy as np


class NetGen:

    def __init__(self):
        pass

    def get_std_net(self, latent_dim=1024) -> K.Sequential:
        encoder_input = K.layers.Input(shape=(
                None, 390))

        encoder = LSTM(latent_dim, return_state=True)
        encoder_output, state_h, state_c = encoder(encoder_input)
        encoder_states = [state_h, state_c]

        decoder_input = K.layers.Input(shape=(None, 1220))
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
        decoder_dense = K.layers.Dense(1220, activation='softmax')
        decoder_output = decoder_dense(decoder_output)

        model = K.models.Model([encoder_input, decoder_input], decoder_output)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        encoder_model = K.models.Model(encoder_input, encoder_states)
        decoder_state_input_h = K.layers.Input(shape=(latent_dim,))
        decoder_state_input_c = K.layers.Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_output, state_h, state_c = decoder_lstm(decoder_input, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_output)
        decoder_model = K.models.Model([decoder_input] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        name = self.generate_netname(latent_dim)
        return model, name, encoder_model, decoder_model

    def generate_netname(self, latent_dim):
        name = ('net_' + str(datetime.datetime.now()) + '_latent_dim_' + str(latent_dim)).replace(' ', '_')
        return name
