import gen_data
import train_net
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



#test_data_generation()
#create_and_save_data('/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/data_as_np_array', 'rwth_corpus')
train(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])

recording_locations = '/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/rwth-phoenix-full-corpus-images'
model_path = '/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/models'
corpus_path = '/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/rwth-phoenix-full-20120323.corpus'
data_location = '/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/data_as_np_array'
data_name = 'rwth_corpus'
batch_size = 50
num_epochs = 1000
save_interval = 100
