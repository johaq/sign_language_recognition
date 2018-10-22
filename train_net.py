import gen_data
import gen_net
import keras as K
import numpy as np
from random import shuffle


class NetTrain:
    def __init__(self, model_path, data_location, data_name, latent_dim=1024, arch="std"):
        self.net_generator = gen_net.NetGen()
        if arch == "std":
            self.model, self.model_name= self.net_generator.get_std_net(latent_dim)
        elif arch == "std_conv":
            self.model, self.model_name= self.net_generator.get_std_conv_net(latent_dim)
        elif arch == "deep_conv":
            self.model, self.model_name= self.net_generator.get_deep_conv_net(latent_dim)
        elif arch == "std_conv_merge":
            self.model, self.model_name= self.net_generator.get_std_conv_merge_net(latent_dim)
        elif arch == "deep_conv_merge":
            self.model, self.model_name= self.net_generator.get_deep_conv_merge_net(latent_dim)
        elif arch == "vgg_19_true":
            self.model, self.model_name= self.net_generator.get_vgg_19_image_net(latent_dim, trainable=True)
        elif arch == "vgg_19_false":
            self.model, self.model_name= self.net_generator.get_vgg_19_image_net(latent_dim, trainable=False)


        self.data_generator = gen_data.DataGen(
            data_path="/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/rwth-phoenix-full-corpus-images/",
            corpus_path="/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/rwth-phoenix-full-20120323.corpus")
        self.data_generator.load_from_file(data_location, data_name)
        #self.data_generator.compute_pca_of_image_set()
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

        indexing = [i for i in range(len(self.data_generator.encoder_input))]
        shuffle(indexing)
        index = 0

        for epoch in range(initial, end):

            encoder_input_data, decoder_input_data, decoder_target_data = self.data_generator.get_sample(indexing[index])
            index += 1
            if index >= len(self.data_generator.encoder_input):
                index = 0
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
                           verbose=1)

        self.model.save(self.path + self.model_name + '.epoch{end:04d}')
        return self.model

    def train_model_images(self,
                    initial: int = 0,
                    end: int = 1,
                    batch_size=1,
                    save_interval = 100,
                    with_op = False) -> K.models.Sequential:

        best_model = K.callbacks.ModelCheckpoint(self.path + self.reduce_modelname(
            self.model_name) + str(with_op) + '.epoch{epoch:04d}',
                                                 save_best_only=True,
                                                 monitor='categorical_accuracy',
                                                 verbose=1)
        checkpoint = K.callbacks.ModelCheckpoint(self.path + self.reduce_modelname(
            self.model_name) + str(with_op) + '.epoch{epoch:04d}',
                                                 verbose=1)

        indexing = [i for i in range(len(self.data_generator.encoder_input))]
        shuffle(indexing)
        index = 0


        for epoch in range(initial, end):

            try:
                if with_op:
                    encoder_input_data, decoder_input_data, decoder_target_data = self.data_generator.get_image_op_sample(indexing[index])
                else:
                    encoder_input_data, decoder_input_data, decoder_target_data = self.data_generator.get_image_sample(indexing[index])
            except:
                index += 1
                print("skipping")
                continue
            index += 1
            if index >= len(self.data_generator.encoder_input):
                index = 0

            if batch_size == 1:
                encoder_input_data = np.expand_dims(encoder_input_data, 0)
                decoder_input_data = np.expand_dims(decoder_input_data, 0)
                decoder_target_data = np.expand_dims(decoder_target_data, 0)
                #encoder_input_data, decoder_input_data, decoder_target_data = self.data_generator.create_batch(batch_size=batch_size)
            else:
                encoder_input_data = self.data_generator.augment_data(encoder_input_data, batch_size)
                decoder_input_data = np.expand_dims(decoder_input_data, 0)
                decoder_input_data_batch = np.concatenate((decoder_input_data, decoder_input_data))
                decoder_target_data = np.expand_dims(decoder_target_data, 0)
                decoder_target_data_batch = np.concatenate((decoder_target_data, decoder_target_data))

                for i in range(batch_size-1):
                    decoder_input_data_batch = np.concatenate((decoder_input_data_batch, decoder_input_data))
                    decoder_target_data_batch = np.concatenate((decoder_target_data_batch, decoder_target_data))
                decoder_input_data = decoder_input_data_batch
                decoder_target_data = decoder_target_data_batch

            callbacks = [best_model]
            if not epoch % save_interval:
                callbacks.append(checkpoint)

            self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                           batch_size=batch_size,
                           callbacks=callbacks,
                           epochs=epoch+1,
                           initial_epoch=epoch,
                           verbose=1)

        self.model.save(self.path + self.model_name + str(with_op) + '.epoch{end:04d}')
        return self.model

    def train_model_mix(self,
                    initial: int = 0,
                    end: int = 1,
                    batch_size=1,
                    save_interval = 100,
                    with_op = False) -> K.models.Sequential:

        best_model = K.callbacks.ModelCheckpoint(self.path + self.reduce_modelname(
            self.model_name) + str(with_op) + '.epoch{epoch:04d}',
                                                 save_best_only=True,
                                                 monitor='categorical_accuracy',
                                                 verbose=1)
        checkpoint = K.callbacks.ModelCheckpoint(self.path + self.reduce_modelname(
            self.model_name) + str(with_op) + '.epoch{epoch:04d}',
                                                 verbose=1)

        indexing = [i for i in range(len(self.data_generator.encoder_input))]
        shuffle(indexing)
        index = 0

        for epoch in range(initial, end):

            try:
                if with_op:
                    encoder_input_data_im, decoder_input_data_im, decoder_target_data_im, encoder_input_data_op= self.data_generator.get_mix_op_sample(indexing[index])
                else:
                    encoder_input_data_im, decoder_input_data_im, decoder_target_data_im, encoder_input_data_op= self.data_generator.get_mix_sample(indexing[index])
            except:
                index += 1
                print("skipping: PIL truncated image error")
                continue

            if encoder_input_data_im.shape[0] != encoder_input_data_op.shape[0]:
                index += 1
                print("skipping: json files incomplete")
                continue

            index += 1
            if index >= len(self.data_generator.encoder_input):
                index = 0

            if batch_size == 1:
                encoder_input_data_im = np.expand_dims(encoder_input_data_im, 0)
                decoder_input_data_im = np.expand_dims(decoder_input_data_im, 0)
                decoder_target_data_im = np.expand_dims(decoder_target_data_im, 0)
                encoder_input_data_op = np.expand_dims(encoder_input_data_op, 0)
            else:
                encoder_input_data_im = self.data_generator.augment_data(encoder_input_data_im, batch_size)
                decoder_input_data_im = np.expand_dims(decoder_input_data_im, 0)
                decoder_input_data_im_batch = np.concatenate((decoder_input_data_im, decoder_input_data_im))
                decoder_target_data_im = np.expand_dims(decoder_target_data_im, 0)
                decoder_target_data_im_batch = np.concatenate((decoder_target_data_im, decoder_target_data_im))
                encoder_input_data_op = np.expand_dims(encoder_input_data_op, 0)
                encoder_input_data_op_batch = np.concatenate((encoder_input_data_op, encoder_input_data_op))

                for i in range(batch_size-1):
                    decoder_input_data_im_batch = np.concatenate((decoder_input_data_im_batch, decoder_input_data_im))
                    decoder_target_data_im_batch = np.concatenate((decoder_target_data_im_batch, decoder_target_data_im))
                    encoder_input_data_op_batch = np.concatenate((encoder_input_data_op_batch, encoder_input_data_op))
                decoder_input_data_im = decoder_input_data_im_batch
                decoder_target_data_im = decoder_target_data_im_batch
                encoder_input_data_op = encoder_input_data_op_batch

            callbacks = [best_model]
            if not epoch % save_interval:
                callbacks.append(checkpoint)

            self.model.fit([encoder_input_data_im, encoder_input_data_op, decoder_input_data_im], decoder_target_data_im,
                           batch_size=batch_size,
                           callbacks=callbacks,
                           epochs=epoch+1,
                           initial_epoch=epoch,
                           verbose=1)

        self.model.save(self.path + self.model_name + str(with_op) + '.epoch{end:04d}')
        return self.model

    def reduce_modelname(self,
                         name: str) -> str:
        if name.split('.')[-1].startswith('epoch'):
            name = '.'.join(name.split('.')[:-1])
            name = name.split('/')[-1]
        return name