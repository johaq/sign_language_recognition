import datetime
from keras.layers import Dense, LSTM, RepeatVector, Flatten, Conv1D, MaxPooling1D, Concatenate, Add
import keras as K
import numpy as np


class NetEval:

    def __init__(self, model_path):
        model = K.models.load_model(model_path)

        encoder_inputs = model.input[0]
        encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = K.models.Model(encoder_inputs, encoder_states)

        decoder_inputs = model.input[1]
        decoder_state_input_h = K.layers.Input(shape=(1024,), name='input3')
        decoder_state_input_c = K.layers.Input(shape=(1024,), name='input4')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = K.models.Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

    # generate target given source sequence
    def predict_sequence(self, source, n_steps=168, cardinality=118):
        # encode
        state = self.encoder_model.predict(source)
        # start of sequence input
        target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
        # collect predictions
        output = list()
        for t in range(n_steps):
            # predict next char
            yhat, h, c = self.decoder_model.predict([target_seq] + state)
            # store prediction
            output.append(yhat[0, 0, :])
            # update state
            state = [h, c]
            # update target sequence
            target_seq = yhat
        print(np.array(output).shape)
        print(np.array(output))
        for i in range(np.array(output).shape[0]):
            decoded_char = self.decode(np.array(output)[i])
            print(chr(decoded_char))
        return np.array(output)

    def decode(self, char):
        return np.argmax(char)