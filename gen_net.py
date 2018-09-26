import datetime
from keras.layers import Dense, LSTM, RepeatVector, Flatten, Conv1D, MaxPooling1D, Concatenate, Add, ConvLSTM2D, Flatten, TimeDistributed
import keras as K
import seq2seq
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
        K.utils.plot_model(model, to_file="model_std.png", show_shapes=True)

        name = self.generate_netname(latent_dim, "stdLSTM")
        return model, name

    def get_std_conv_net(self, latent_dim=1024) -> K.Sequential:
        image_input = K.layers.Input(shape=(
                None, 260, 210, 3))

        image_conv = K.layers.TimeDistributed(K.layers.Conv2D(filters=latent_dim, kernel_size=(4, 4)), name="bla")
        image_conv_out = image_conv(image_input)

        image_max = K.layers.TimeDistributed(K.layers.MaxPool2D(pool_size=(2, 2)))
        image_max_out = image_max(image_conv_out)

        image_flat = K.layers.TimeDistributed(K.layers.Flatten())
        encoder_input = image_flat(image_max_out)

        encoder = LSTM(latent_dim, return_state=True)
        encoder_output, state_h, state_c = encoder(encoder_input)
        encoder_states = [state_h, state_c]

        decoder_input = K.layers.Input(shape=(None, 1220))
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
        decoder_dense = K.layers.Dense(1220, activation='softmax')
        decoder_output = decoder_dense(decoder_output)

        model = K.models.Model([image_input, decoder_input], decoder_output)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        K.utils.plot_model(model, to_file="model.png", show_shapes=True)
        model.summary()

        name = self.generate_netname(latent_dim, "convLSTM")
        return model, name

    def get_simple_seq2seq_net(self):
        pass

    def get_adv_seq2seq_net(self):
        pass

    def get_adv_seq2seq_net_with_attention(self):
        pass

    def generate_netname(self, latent_dim, type):
        name = ('net_' + str(datetime.datetime.now()) + '_' + type + '_latent_dim_' + str(latent_dim)).replace(' ', '_')
        return name
