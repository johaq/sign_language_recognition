import datetime
from keras.layers import Dense, LSTM, RepeatVector, Flatten, Conv1D, MaxPooling1D, Concatenate, Add
import keras as K
import numpy as np
import editdistance

class NetEval:

    def __init__(self, model, dict, load_model, latent_dim, mix=False):
        if load_model:
            model = K.models.load_model(model)
            K.utils.plot_model(model, to_file="eval_net.png", show_shapes=True)
        self.dict = dict

        if not mix:
            self.mix = False
            encoder_inputs = model.input[0]
        else:
            self.mix = True
            encoder_inputs = [model.input[0], model.input[1]]
        encoder_outputs, state_h_enc, state_c_enc = model.layers[-3].output
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = K.models.Model(encoder_inputs, encoder_states)

        if not mix:
            decoder_inputs = model.input[1]
        else:
            decoder_inputs = model.input[2]
        decoder_state_input_h = K.layers.Input(shape=(latent_dim,), name='altinput1')
        decoder_state_input_c = K.layers.Input(shape=(latent_dim,), name='altinput2')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.layers[-2]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = model.layers[-1]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = K.models.Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

    # generate target given source sequence
    def predict_sequence(self, source, n_steps=168, cardinality=1220):
        # encode
        if self.mix:
            state = self.encoder_model.predict([source[0], source[1]])
        else:
            state = self.encoder_model.predict(source)
        # start of sequence input
        target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
        # collect predictions
        output = list()
        prediction_string = ""
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
            for x in self.dict:
                if self.dict[x] == decoded_word:
                    print(x, end=' ')
                    prediction_string = prediction_string + x
        return np.array(output), prediction_string

    def decode(self, datum):
        return np.argmax(datum)

    def test(self, encoder_input_test, decoder_output_test):
        acc_total = 0
        for i in range(len(encoder_input_test)):
            print("\n Target:")
            for j in range(decoder_output_test[i].shape[0]):
                decoded_word = self.decode(decoder_output_test[i][j])
                for x in self.dict:
                    if self.dict[x] == decoded_word:
                        print(x, end=' ')
            prediction = self.predict_sequence(np.expand_dims(encoder_input_test[i], 0), n_steps=decoder_output_test[i].shape[0])
            acc = 0
            for j in range(len(prediction)):
                if self.decode(prediction[j]) == self.decode(decoder_output_test[i][j]):
                    acc += 1
            acc = acc / len(prediction)
            acc_total += acc
        return acc_total / len(encoder_input_test)

    def test_edit_distance(self, encoder_input_test, decoder_output_test):
        if self.mix:
            encoder_input_test_op = encoder_input_test[1]
            encoder_input_test = encoder_input_test[0]
        acc_total = 0
        target_string = ""
        for i in range(len(encoder_input_test)):
            print("\n Target:")
            for j in range(decoder_output_test[i].shape[0]):
                decoded_word = self.decode(decoder_output_test[i][j])
                for x in self.dict:
                    if self.dict[x] == decoded_word:
                        print(x, end=' ')
                        target_string = target_string + x
            if self.mix:
                prediction, prediction_string = self.predict_sequence([np.expand_dims(encoder_input_test[i], 0),np.expand_dims(encoder_input_test_op[i], 0)],
                                               n_steps=decoder_output_test[i].shape[0])
            else:
                prediction, prediction_string = self.predict_sequence(
                    np.expand_dims(encoder_input_test[i], 0),
                    n_steps=decoder_output_test[i].shape[0])
            acc = 1 - editdistance.eval(prediction_string, target_string) / np.maximum(len(prediction_string), len(target_string))
            acc_total += acc
        return acc_total / len(encoder_input_test)