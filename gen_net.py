import datetime
from keras.layers import Dense, LSTM, RepeatVector, Flatten, Conv1D, MaxPooling1D, Concatenate, Add, ConvLSTM2D, Flatten, TimeDistributed
import keras as K
import seq2seq
from seq2seq.models import Seq2Seq
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

        image_conv = K.layers.TimeDistributed(K.layers.Conv2D(filters=3, kernel_size=(10, 10)))
        image_conv_out = image_conv(image_input)

        image_max = K.layers.TimeDistributed(K.layers.MaxPool2D(pool_size=(5, 5)))
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

    def get_deep_conv_net(self, latent_dim=1024) -> K.Sequential:
        image_input = K.layers.Input(shape=(
                None, 260, 210, 3))

        image_conv_1 = K.layers.TimeDistributed(K.layers.Conv2D(filters=3, kernel_size=(5, 5)))
        image_conv_out_1 = image_conv_1(image_input)

        image_max_1 = K.layers.TimeDistributed(K.layers.MaxPool2D(pool_size=(2, 2)))
        image_max_out_1 = image_max_1(image_conv_out_1)

        image_conv_2 = K.layers.TimeDistributed(K.layers.Conv2D(filters=3, kernel_size=(5, 5)))
        image_conv_out_2 = image_conv_2(image_max_out_1)

        image_max_2 = K.layers.TimeDistributed(K.layers.MaxPool2D(pool_size=(2, 2)))
        image_max_out_2 = image_max_2(image_conv_out_2)

        image_conv_3 = K.layers.TimeDistributed(K.layers.Conv2D(filters=3, kernel_size=(5, 5)))
        image_conv_out_3 = image_conv_3(image_max_out_2)

        image_max_3 = K.layers.TimeDistributed(K.layers.MaxPool2D(pool_size=(2, 2)))
        image_max_out_3 = image_max_3(image_conv_out_3)

        image_conv_4 = K.layers.TimeDistributed(K.layers.Conv2D(filters=3, kernel_size=(5, 5)))
        image_conv_out_4 = image_conv_4(image_max_out_3)

        image_max_4 = K.layers.TimeDistributed(K.layers.MaxPool2D(pool_size=(2, 2)))
        image_max_out_4 = image_max_4(image_conv_out_4)

        image_flat = K.layers.TimeDistributed(K.layers.Flatten())
        encoder_input = image_flat(image_max_out_4)

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

        name = self.generate_netname(latent_dim, "deepConvLSTM")
        return model, name

    def get_deep_conv_net_google(self, latent_dim=1024) -> K.Sequential:
        image_input = K.layers.Input(shape=(
                None, 260, 210, 3))

        image_conv_1 = K.layers.TimeDistributed(K.layers.Conv2D(filters=3, kernel_size=(3, 3)))
        image_conv_out_1 = image_conv_1(image_input)

        image_max_1 = K.layers.TimeDistributed(K.layers.MaxPool2D(pool_size=(2, 2)))
        image_max_out_1 = image_max_1(image_conv_out_1)

        image_conv_2 = K.layers.TimeDistributed(K.layers.Conv2D(filters=3, kernel_size=(1, 1)))
        image_conv_out_2 = image_conv_2(image_max_out_1)

        image_max_2 = K.layers.TimeDistributed(K.layers.MaxPool2D(pool_size=(2, 2)))
        image_max_out_2 = image_max_2(image_conv_out_2)

        image_conv_3 = K.layers.TimeDistributed(K.layers.Conv2D(filters=3, kernel_size=(3, 3)))
        image_conv_out_3 = image_conv_3(image_max_out_2)

        image_max_3 = K.layers.TimeDistributed(K.layers.MaxPool2D(pool_size=(2, 2)))
        image_max_out_3 = image_max_3(image_conv_out_3)

        image_conv_4 = K.layers.TimeDistributed(K.layers.Conv2D(filters=3, kernel_size=(3, 3)))
        image_conv_out_4 = image_conv_4(image_max_out_3)

        image_max_4 = K.layers.TimeDistributed(K.layers.MaxPool2D(pool_size=(2, 2)))
        image_max_out_4 = image_max_4(image_conv_out_4)

        image_flat = K.layers.TimeDistributed(K.layers.Flatten())
        encoder_input = image_flat(image_max_out_4)

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

        name = self.generate_netname(latent_dim, "deepConvLSTM")
        return model, name

    def get_std_conv_merge_net(self, latent_dim=1024) -> K.Sequential:
        image_input = K.layers.Input(shape=(
                None, 260, 210, 3))

        feature_input = K.layers.Input(shape=(
            None, 390))

        image_conv = K.layers.TimeDistributed(K.layers.Conv2D(filters=3, kernel_size=(10, 10)), name="bla")
        image_conv_out = image_conv(image_input)

        image_max = K.layers.TimeDistributed(K.layers.MaxPool2D(pool_size=(5, 5)))
        image_max_out = image_max(image_conv_out)

        image_flat = K.layers.TimeDistributed(K.layers.Flatten())
        image_processed_input = image_flat(image_max_out)

        concat_layer = K.layers.Concatenate(axis=-1)
        encoder_input = concat_layer([image_processed_input, feature_input])

        encoder = LSTM(latent_dim, return_state=True)
        encoder_output, state_h, state_c = encoder(encoder_input)
        encoder_states = [state_h, state_c]

        decoder_input = K.layers.Input(shape=(None, 1220))
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
        decoder_dense = K.layers.Dense(1220, activation='softmax')
        decoder_output = decoder_dense(decoder_output)

        model = K.models.Model([image_input, feature_input, decoder_input], decoder_output)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        K.utils.plot_model(model, to_file="model.png", show_shapes=True)
        model.summary()

        name = self.generate_netname(latent_dim, "convMergeLSTM")
        return model, name

    def get_deep_conv_merge_net(self, latent_dim=1024) -> K.Sequential:
        image_input = K.layers.Input(shape=(
                None, 260, 210, 3))

        feature_input = K.layers.Input(shape=(
            None, 390))

        image_conv_1 = K.layers.TimeDistributed(K.layers.Conv2D(filters=3, kernel_size=(5, 5)))
        image_conv_out_1 = image_conv_1(image_input)

        image_max_1 = K.layers.TimeDistributed(K.layers.MaxPool2D(pool_size=(2, 2)))
        image_max_out_1 = image_max_1(image_conv_out_1)

        image_conv_2 = K.layers.TimeDistributed(K.layers.Conv2D(filters=3, kernel_size=(5, 5)))
        image_conv_out_2 = image_conv_2(image_max_out_1)

        image_max_2 = K.layers.TimeDistributed(K.layers.MaxPool2D(pool_size=(2, 2)))
        image_max_out_2 = image_max_2(image_conv_out_2)

        image_conv_3 = K.layers.TimeDistributed(K.layers.Conv2D(filters=3, kernel_size=(5, 5)))
        image_conv_out_3 = image_conv_3(image_max_out_2)

        image_max_3 = K.layers.TimeDistributed(K.layers.MaxPool2D(pool_size=(2, 2)))
        image_max_out_3 = image_max_3(image_conv_out_3)

        image_conv_4 = K.layers.TimeDistributed(K.layers.Conv2D(filters=3, kernel_size=(5, 5)))
        image_conv_out_4 = image_conv_4(image_max_out_3)

        image_max_4 = K.layers.TimeDistributed(K.layers.MaxPool2D(pool_size=(2, 2)))
        image_max_out_4 = image_max_4(image_conv_out_4)

        image_flat = K.layers.TimeDistributed(K.layers.Flatten())
        image_processed_input = image_flat(image_max_out_4)

        concat_layer = K.layers.Concatenate(axis=-1)
        encoder_input = concat_layer([image_processed_input, feature_input])

        encoder = LSTM(latent_dim, return_state=True)
        encoder_output, state_h, state_c = encoder(encoder_input)
        encoder_states = [state_h, state_c]

        decoder_input = K.layers.Input(shape=(None, 1220))
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
        decoder_dense = K.layers.Dense(1220, activation='softmax')
        decoder_output = decoder_dense(decoder_output)

        model = K.models.Model([image_input, feature_input, decoder_input], decoder_output)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        K.utils.plot_model(model, to_file="model.png", show_shapes=True)
        model.summary()

        name = self.generate_netname(latent_dim, "deepConvMergeLSTM")
        return model, name

    def get_simple_seq2seq_net(self):
        pass

    def get_adv_seq2seq_net(self):
        #model = Seq2Seq(batch_input_shape=(16, 7, 5), hidden_dim=10, output_length=8, output_dim=20, depth=4, peek=True)
        #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        pass

    def get_adv_seq2seq_net_with_attention(self):
        pass

    def generate_netname(self, latent_dim, type):
        name = ('net_' + str(datetime.datetime.now()) + '_' + type + '_latent_dim_' + str(latent_dim)).replace(' ', '_')
        return name
