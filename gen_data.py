import json
import numpy as np
import os
import xml.etree.ElementTree as ET
import random
import keras as K



class DataGen:

    def __init__(self, data_path, corpus_path):
        self.data_path = data_path
        self.encoder_input = []
        self.decoder_input = []
        self.decoder_output = []
        self.corpus_path = corpus_path

    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def read_json(self, filename):
        json_data=open(filename).read()
        data = json.loads(json_data)
        return data

    def json_to_train_data(self, json_data):
        # read in keypoints from json file
        if len(json_data["people"]) == 0:
            raise Exception("People list empty")
        hand_left_keypoints_2d = json_data["people"][0]["hand_left_keypoints_2d"] #catch exception if no people in file
        hand_right_keypoints_2d = json_data["people"][0]["hand_right_keypoints_2d"]
        pose_keypoints_2d = json_data["people"][0]["pose_keypoints_2d"]
        face_keypoints_2d = json_data["people"][0]["face_keypoints_2d"]

        # make 3xl matrices with [keypoint_x, keypoint_y, confidence] where l is number of bodyparts
        hand_left_length = 21
        hand_left_keypoints_2d_m = [hand_left_keypoints_2d[0+(3*x):3+(3*x)] for x in range(hand_left_length)]
        hand_right_length = 21
        hand_right_keypoints_2d_m = [hand_right_keypoints_2d[0 + (3 * x):3 + (3 * x)] for x in range(hand_right_length)]
        pose_keypoints_length = 18
        pose_keypoints_2d_m = [pose_keypoints_2d[0 + (3 * x):3 + (3 * x)] for x in range(pose_keypoints_length)]
        face_keypoints_length = 70
        face_keypoints_2d_m = [face_keypoints_2d[0 + (3 * x):3 + (3 * x)] for x in range(face_keypoints_length)]

        x_norm = pose_keypoints_2d_m[1][0] # x position of neck
        y_norm = pose_keypoints_2d_m[1][1] # y position of neck

        x_max = 210 # x dim of image
        y_max = 260 # y dim of image

        # normalize around neck point, confidence already normalized
        hand_left_keypoints_2d_m_norm = [
            [(hand_left_keypoints_2d_m[x][0] - x_norm) / x_max, (hand_left_keypoints_2d_m[x][1] - y_norm) / y_max,
             hand_left_keypoints_2d_m[x][2]] for x in range(hand_left_length)]
        hand_right_keypoints_2d_m_norm = [
            [(hand_right_keypoints_2d_m[x][0] - x_norm) / x_max, (hand_right_keypoints_2d_m[x][1] - y_norm) / y_max,
             hand_right_keypoints_2d_m[x][2]] for x in range(hand_right_length)]
        pose_keypoints_2d_m_norm = [
            [(pose_keypoints_2d_m[x][0] - x_norm) / x_max, (pose_keypoints_2d_m[x][1] - y_norm) / y_max,
             pose_keypoints_2d_m[x][2]] for x in range(pose_keypoints_length)]
        face_keypoints_2d_m_norm = [
            [(face_keypoints_2d_m[x][0] - x_norm) / x_max, (face_keypoints_2d_m[x][1] - y_norm) / y_max,
             face_keypoints_2d_m[x][2]] for x in range(face_keypoints_length)]

        # stack matrices to one feature
        feature_m = np.row_stack((np.row_stack((pose_keypoints_2d_m_norm, face_keypoints_2d_m_norm)),
                                  np.row_stack((hand_left_keypoints_2d_m_norm, hand_right_keypoints_2d_m_norm))))
        return feature_m.flatten()

    def read_recording(self, name):
        files = os.listdir(name + "/openpose/")
        first = True
        for f in files:
            if f.endswith(".json"):
                data = self.read_json(name + "/openpose/" + f)
                try:
                    feature = self.json_to_train_data(data)
                    if first:
                        feature_tensor = np.expand_dims(feature, 0)
                        first = False
                    else:
                        feature = np.expand_dims(feature, 0)
                        feature_tensor = np.concatenate((feature_tensor, feature))
                except:
                    print("ERROR: Could not create feature from json")
                    pass
        # feature_tensor normalized
        return feature_tensor

    def read_label(self, name):
        tree = ET.parse(self.corpus_path)
        root = tree.getroot()
        for recording in root.findall('recording'):
            if recording.get('name') == name:
                seg = recording.find('segment')
                orth = seg.find('orth').text
                orth = orth.strip()
                # integer encoding
                int_encoded_input = [ord(char) for char in " " + orth[:-1] ]
                int_encoded_output = [ord(char) for char in orth]
                # one-hot encoding
                one_hot_encoded_input = K.utils.to_categorical(int_encoded_input, num_classes=118)
                one_hot_encoded_output = K.utils.to_categorical(int_encoded_output, num_classes=118)
                # offset label by one timestep for prediction
                return one_hot_encoded_input, one_hot_encoded_output


    def read_path(self):
        recordings = os.listdir(self.data_path)
        encoder_input = []
        decoder_input = []
        decoder_output = []
        c = 0
        for r in recordings:
            if not r.startswith("."):
                recording = self.read_recording(self.data_path + "/" + r)
                encoder_input.append(recording)
                new_decoder_input, new_decoder_output = self.read_label(r)
                decoder_input.append(new_decoder_input)
                decoder_output.append(new_decoder_output)
                c += 1
                if c % 100 == 0:
                    print(str(c) + "/" + str(len(recordings)))

        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.decoder_output = decoder_output
        return encoder_input, decoder_input, decoder_output

    def create_batch(self, batch_size):
        index_list = random.sample(range(len(self.encoder_input)), batch_size)

        batch_data_en = [self.e_input_padded[i] for i in index_list]
        batch_data_en_arr = np.expand_dims(batch_data_en[0], 0)
        for d in range(1, len(batch_data_en)):
            batch_data_en_arr = np.concatenate((batch_data_en_arr, np.expand_dims(batch_data_en[d], 0)))

        batch_de_in = [self.d_input_padded[i] for i in index_list]
        batch_de_in_arr = np.expand_dims(batch_de_in[0], 0)
        for l in range(1,len(batch_de_in)):
            batch_de_in_arr = np.concatenate((batch_de_in_arr, np.expand_dims(batch_de_in[l], 0)))

        batch_de_out = [self.d_output_padded[i] for i in index_list]
        batch_de_out_arr = np.expand_dims(batch_de_out[0], 0)
        for l in range(1, len(batch_de_out)):
            batch_de_out_arr = np.concatenate((batch_de_out_arr, np.expand_dims(batch_de_out[l], 0)))

        return batch_data_en_arr, batch_de_in_arr, batch_de_out_arr

    def get_random_sample(self):
        index = random.randint(0, len(self.encoder_input) - 1)
        return self.encoder_input[index], self.decoder_input[index], self.decoder_output[index]

    def load_from_file(self, path, filename):
        self.encoder_input = np.load(path + '/' + '' + filename + '_' + 'e_input.npy')
        self.decoder_input = np.load(path + '/' + '' + filename + '_' + 'd_input.npy')
        self.decoder_output = np.load(path + '/' + '' + filename + '_' + 'd_output.npy')
        # Pad input for Batch computation
        self.e_input_padded = K.preprocessing.sequence.pad_sequences(self.encoder_input, maxlen=None, dtype='float32',
                                                                padding='pre',
                                                                truncating='pre', value=0.0)
        self.d_input_padded = K.preprocessing.sequence.pad_sequences(self.decoder_input, maxlen=None, dtype='float32',
                                                                padding='pre',
                                                                truncating='pre', value=0.0)
        self.d_output_padded = K.preprocessing.sequence.pad_sequences(self.decoder_output, maxlen=None, dtype='float32',
                                                                padding='pre',
                                                                truncating='pre', value=0.0)
        print(self.e_input_padded.shape)
        print(self.d_input_padded.shape)
        print(self.d_output_padded.shape)




#g_data = read_json("/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/rwth-phoenix-full-corpus-images/01April_2010_Thursday_heute_default-0/openpose/01April_2010_Thursday_heute.avi_fn044294-0_keypoints.json")
#g_feature = json_to_train_data(g_data)
#print("Extracted Feature")
