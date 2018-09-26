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
    dict = data_generator.create_dictionary()
    np.save(
        path + '/' + '' + filename + '_' + 'dict.npy',
        dict)
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


def create_and_save_data_images(path, filename):
    data_generator = gen_data.DataGen(
        data_path='/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/rwth-phoenix-full-corpus-images',
        corpus_path='/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/rwth-phoenix-full-20120323.corpus')
    dict = data_generator.create_dictionary()
    np.save(
        path + '/' + '' + filename + '_' + 'dict.npy',
        dict)
    e_input, d_input, d_output = data_generator.read_path_images()
    np.save(
        path + '/' + '' + filename + '_' + 'e_input.npy',
        e_input)
    np.save(
        path + '/' + '' + filename + '_' + 'd_input.npy',
        d_input)
    np.save(
        path + '/' + '' + filename + '_' + 'd_output.npy',
        d_output)


def train(model_path, data_location, data_name, batch_size, num_epochs, save_interval, latent_dim):
    net_trainer = train_net.NetTrain(model_path, data_location, data_name, latent_dim)
    model_trained = net_trainer.train_model_images(batch_size=batch_size, end=num_epochs, save_interval=save_interval)
    return net_trainer, model_trained


def evaluate(model, dict, latent_dim, encoder_input_data, decoder_output_data, load_model=True,):
    net_eval = eval_net.NetEval(model, dict, load_model, latent_dim)
    return net_eval.test(encoder_input_data, decoder_output_data)


#test_data_generation()
#create_and_save_data_images('/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/data_as_np_array', 'rwth_corpus_images')

print('########### START CONV TRAINING WITH DATA_NAME:%s NUM_EPOCHS:%d LATENT_DIM:%d ###########' % (sys.argv[3], int(sys.argv[5]), int(sys.argv[7])))
trainer, model = train(model_path=sys.argv[1], data_location=sys.argv[2], data_name=sys.argv[3],
                       batch_size=int(sys.argv[4]), num_epochs=int(sys.argv[5]),
                       save_interval=int(sys.argv[6]), latent_dim=int(sys.argv[7]))
print('########### EVALUATE MODEL ###########')
images = True
if images:
    acc = 0
    for i in range(100):
        encoder_input_data, decoder_input_data, decoder_target_data = trainer.data_generator.get_random_image_sample()
        acc += evaluate(model, trainer.data_generator.dict, int(sys.argv[7]),
                   np.expand_dims(encoder_input_data, 0), np.expand_dims(decoder_target_data, 0), load_model=False)
    acc = acc / 100
else:
    acc = evaluate(model, trainer.data_generator.dict, int(sys.argv[7]),
                   trainer.data_generator.encoder_input_test, trainer.data_generator.decoder_output_test,
                   load_model=False)
print('\n########### MODEL ACCURACY: %f ###########' % acc)