import gen_data
import gen_net
import keras as K
import numpy as np


class NetTrain:
    def __init__(self, model_path, data_location, data_name):
        self.net_generator = gen_net.NetGen()
        self.model, self.model_name, self.encoder_model, self.decoder_model = self.net_generator.get_std_net()
        self.data_generator = gen_data.DataGen(data_path="", corpus_path="")
        self.data_generator.load_from_file(data_location, data_name)
        self.path = model_path

    def train_model(self,
                    initial: int = 0,
                    end: int = 1,
                    batch_size=1,
                    save_interval = 100) -> K.models.Sequential:

        best_model = K.callbacks.ModelCheckpoint(self.path + self.reduce_modelname(
            self.model_name) + '.epoch{epoch:04d}',
                                                 save_best_only=True,
                                                 monitor='categorical_accuracy',
                                                 verbose=1)
        checkpoint = K.callbacks.ModelCheckpoint(self.path + self.reduce_modelname(
            self.model_name) + '.epoch{epoch:04d}',
                                                 verbose=1)

        for epoch in range(initial, end):
            encoder_input_data, decoder_input_data, decoder_target_data = self.data_generator.create_batch(batch_size=batch_size)

            callbacks = [best_model]
            if not epoch % save_interval:
                callbacks.append(checkpoint)

            self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                           batch_size=batch_size,
                           callbacks=callbacks,
                           epochs=epoch+1,
                           initial_epoch=epoch,
                           verbose=1)

        self.model.save(self.path + self.model_name)
        self.predict_sequence(encoder_input_data[0])
        return self.model

    def reduce_modelname(self,
                         name: str) -> str:
        if name.split('.')[-1].startswith('epoch'):
            name = '.'.join(name.split('.')[:-1])
            name = name.split('/')[-1]
        return name

    # generate target given source sequence
    def predict_sequence(self, source, n_steps = 299, cardinality = 118):
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
        print(np.array(output))
        return np.array(output)