import datetime
from keras.layers import Dense, LSTM, RepeatVector, Flatten, Conv1D, MaxPooling1D, Concatenate, Add
import keras as K
import numpy as np


class NetEval:

    def __init__(self, model_path, dict):
        model = K.models.load_model(model_path)
        self.dict = dict

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
    def predict_sequence(self, source, n_steps=168, cardinality=1220):
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
        print("\n Prediction:")
        for i in range(np.array(output).shape[0]):
            decoded_word = self.decode(np.array(output)[i])
            print(chr(decoded_word),end='')
        return np.array(output)

    def decode(self, w):
        return self.dict[w]

    def test(self, encoder_input_test, decoder_input_test, decoder_output_test):
        acc_total = 0
        for i in range(len(encoder_input_test)):
            predicition = self.predict_sequence(encoder_input_test[i], n_steps=decoder_output_test[i].shape[1])
            acc = 0
            for j in predicition:
                if predicition[j] == decoder_output_test[j]:
                    acc += 1
            acc = acc / len(predicition)
            acc_total += acc
        return acc_total / len(encoder_input_test)