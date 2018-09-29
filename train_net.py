import gen_data
import gen_net
import keras as K
import numpy as np


class NetTrain:
    def __init__(self, model_path, data_location, data_name, latent_dim=1024):
        self.net_generator = gen_net.NetGen()
        self.model, self.model_name= self.net_generator.get_std_conv_net(latent_dim)
        self.data_generator = gen_data.DataGen(
            data_path="/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/rwth-phoenix-full-corpus-images/",
            corpus_path="/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/rwth-phoenix-full-20120323.corpus")
        self.data_generator.load_from_file(data_location, data_name)
        #self.data_generator.load_from_file(data_location, data_name)
        #self.data_generator.split_testset(0.1)
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

            encoder_input_data, decoder_input_data, decoder_target_data = self.data_generator.get_random_sample()
            encoder_input_data = np.expand_dims(encoder_input_data, 0)
            decoder_input_data = np.expand_dims(decoder_input_data, 0)
            decoder_target_data = np.expand_dims(decoder_target_data, 0)
            #encoder_input_data, decoder_input_data, decoder_target_data = self.data_generator.create_batch(batch_size=batch_size)

            callbacks = [best_model]
            if not epoch % save_interval:
                callbacks.append(checkpoint)

            self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                           batch_size=batch_size,
                           callbacks=callbacks,
                           epochs=epoch+1,
                           initial_epoch=epoch,
                           verbose=0)

        self.model.save(self.path + self.model_name + '.epoch{end:04d}')
        return self.model

    def train_model_images(self,
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

            encoder_input_data, decoder_input_data, decoder_target_data = self.data_generator.get_random_image_sample()
            encoder_input_data = np.expand_dims(encoder_input_data, 0)
            decoder_input_data = np.expand_dims(decoder_input_data, 0)
            decoder_target_data = np.expand_dims(decoder_target_data, 0)
            #encoder_input_data, decoder_input_data, decoder_target_data = self.data_generator.create_batch(batch_size=batch_size)

            callbacks = [best_model]
            if not epoch % save_interval:
                callbacks.append(checkpoint)

            self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                           batch_size=batch_size,
                           callbacks=callbacks,
                           epochs=epoch+1,
                           initial_epoch=epoch,
                           verbose=0)

        self.model.save(self.path + self.model_name + '.epoch{end:04d}')
        return self.model

    def reduce_modelname(self,
                         name: str) -> str:
        if name.split('.')[-1].startswith('epoch'):
            name = '.'.join(name.split('.')[:-1])
            name = name.split('/')[-1]
        return name