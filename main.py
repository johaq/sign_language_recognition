import gen_data
import train_net
import eval_net
import numpy as np
import sys


def test_data_generation():
    data_generator = gen_data.DataGen(data_path='/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/rwth-phoenix-full-corpus-images',
                     corpus_path='/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/rwth-phoenix-full-20120323.corpus')
    e_input, d_input, d_output = data_generator.read_path()
    print("Encoder Input")
    print(e_input[0].shape)
    print(e_input[0])
    print("Decoder Input")
    print(d_input[0].shape)
    print(d_input[0])
    print("Decoder Output")
    print(d_output[0].shape)
    print(d_output[0])
    e_input_b, d_input_b, d_output_b = data_generator.create_batch(50)
    print("Encoder Input Batch")
    print(e_input_b.shape)
    print("Decoder Input")
    print(d_input_b.shape)
    print("Decoder Output")
    print(d_output_b.shape)

def create_and_save_data(path, filename):
    data_generator = gen_data.DataGen(
        data_path='/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/rwth-phoenix-full-corpus-images',
        corpus_path='/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/rwth-phoenix-full-20120323.corpus')
    e_input, d_input, d_output = data_generator.read_path()
    np.save(
        path + '/' + '' + filename + '_' + 'e_input.npy',
        e_input)
    np.save(
        path + '/' + '' + filename + '_' + 'd_input.npy',
        d_input)
    np.save(
        path + '/' + '' + filename + '_' + 'd_output.npy',
        d_output)


def train(model_path, data_location, data_name, batch_size, num_epochs, save_interval):
    net_trainer = train_net.NetTrain(model_path, data_location, data_name)
    model_trained = net_trainer.train_model(batch_size=batch_size, end=num_epochs, save_interval=save_interval)


def evaluate(model_path):
    data_generator = gen_data.DataGen(data_path="", corpus_path="")
    data_generator.load_from_file('/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/data_as_np_array/', "rwth_corpus")

    encoder_input_data, decoder_input_data, decoder_target_data = data_generator.create_batch(1)
    print(decoder_target_data.shape)
    print(decoder_target_data)
    for i in range(decoder_target_data.shape[1]):
        print(chr(np.argmax(decoder_target_data[0][i])))
    net_eval = eval_net.NetEval(model_path)
    net_eval.predict_sequence(encoder_input_data)


#test_data_generation()
#create_and_save_data('/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/data_as_np_array', 'rwth_corpus')
#train(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
#evaluate('/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/models/net_2018-09-14_13:21:30.469090.epoch0001')
recording_locations = '/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/rwth-phoenix-full-corpus-images'
#model_path = '/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/models'
#corpus_path = '/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/rwth-phoenix-full-20120323.corpus'
