import gen_data
import gen_data_signum
import train_net
import train_net_signum
import eval_net
import numpy as np
import sys
import editdistance
import os
import vgg_19_model
import PIL
import pickle
import keras as K
import cmu_model




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

def create_and_save_data_sig(path, filename):
    data_generator = gen_data_signum.DataGenSIGNUM(
        data_path='/media/compute/homes/jkummert/SIGNUM/',
        corpus_path='')
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


def train(model_path, data_location, data_name, batch_size, num_epochs, save_interval, latent_dim, arch="std", with_op=False):
    net_trainer = train_net.NetTrain(model_path, data_location, data_name, latent_dim, arch)
    if arch == "std" :
        model_trained = net_trainer.train_model(batch_size=batch_size, end=num_epochs, save_interval=save_interval)
    elif arch == "std_conv" or arch == "deep_conv" or arch == "vgg_19_true" or arch == "vgg_19_false":
        model_trained = net_trainer.train_model_images(batch_size=batch_size, end=num_epochs, save_interval=save_interval, with_op=with_op)
    elif arch == "std_conv_merge" or arch == "deep_conv_merge":
        model_trained = net_trainer.train_model_mix(batch_size=batch_size, end=num_epochs, save_interval=save_interval, with_op=with_op)
    return net_trainer, model_trained


def train_sig(model_path, data_location, data_name, batch_size, num_epochs, save_interval, latent_dim, arch="std", with_op=False):
    net_trainer = train_net_signum.NetTrain(model_path, data_location, data_name, latent_dim, arch)
    if arch == "std" :
        model_trained = net_trainer.train_model(batch_size=batch_size, end=num_epochs, save_interval=save_interval)
    elif arch == "std_conv" or arch == "deep_conv" or arch == "vgg_19_true" or arch == "vgg_19_false":
        model_trained = net_trainer.train_model_images(batch_size=batch_size, end=num_epochs, save_interval=save_interval, with_op=with_op)
    elif arch == "std_conv_merge" or arch == "deep_conv_merge":
        model_trained = net_trainer.train_model_mix(batch_size=batch_size, end=num_epochs, save_interval=save_interval, with_op=with_op)
    return net_trainer, model_trained



def evaluate(model, dict, latent_dim, encoder_input_data, decoder_output_data, load_model=True, mix=False, num=250):
    net_eval = eval_net.NetEval(model, dict, load_model, latent_dim, mix=mix)
    acc = 0
    for i in range(num):
        acc += net_eval.test_edit_distance(encoder_input_data, decoder_output_data)
    acc = acc/num
    return acc


def evaluate_image_model(model, arch, op, latent_dim, num):
    data_generator = gen_data.DataGen(
        data_path="/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/rwth-phoenix-full-corpus-images/",
        corpus_path="/home/johannes/Documents/master_data/jkummert_master_thesis/rwth/rwth-phoenix-full-20120323.corpus")
    data_generator.load_from_file(sys.argv[2], sys.argv[3])
    net_eval = eval_net.NetEval(model, data_generator.dict, load_model=True, latent_dim=latent_dim, mix=("merge" in arch))

    acc = 0
    for i in range(num):
        try:
            if arch == "std":
                encoder_input_data, decoder_input_data, decoder_target_data = data_generator.get_random_sample()
                encoder_input_data = np.expand_dims(encoder_input_data, 0)
            elif arch == "std_conv" or arch == "deep_conv" or arch == "vgg_19_retrain_False" or arch == "vgg_19_retrain_True":
                if not op:
                    encoder_input_data, decoder_input_data, decoder_target_data = data_generator.get_random_image_sample()
                else:
                    encoder_input_data, decoder_input_data, decoder_target_data = data_generator.get_random_image_op_sample()
                encoder_input_data = np.expand_dims(encoder_input_data, 0)
            elif arch == "std_conv_merge" or arch == "deep_conv_merge":
                if not op:
                    encoder_input_data, decoder_input_data, decoder_target_data, encoder_input_op = data_generator.get_random_mix_sample()
                else:
                    encoder_input_data, decoder_input_data, decoder_target_data, encoder_input_op = data_generator.get_random_mix_op_sample()
                if not is_same_input_size(encoder_input_data, encoder_input_op):
                    continue
                encoder_input_data = [np.expand_dims(encoder_input_data, 0), np.expand_dims(encoder_input_op, 0)]

            acc += net_eval.test_edit_distance(encoder_input_data, np.expand_dims(decoder_target_data, 0))
        except ValueError or OSError:
            continue
    return acc / num


def is_same_input_size(in_im, in_op):
    return in_im.shape[0] == in_op.shape[0]


def decode(datum):
    return np.argmax(datum)


def get_class_distribution():
    net_trainer = train_net_signum.NetTrain(sys.argv[1], sys.argv[2], sys.argv[3], 1, sys.argv[8])
    dist = net_trainer.data_generator.get_class_distribution()
    print(dist)
    return dist, net_trainer


def get_dumb_model_acc():
    dist, trainer = get_class_distribution()
    sum = 0
    for v in dist:
        sum += dist[v]
    print(sum)
    probs = []
    keys = []
    for v in dist:
        dist[v] = dist[v] / sum
        probs.append(dist[v])
        keys.append(v)
    print(dist)
    print(keys)
    print(probs)

    acc_total = 0
    num = 100000
    for i in range(num):
        encoder_input_data, decoder_input_data, decoder_target_data = trainer.data_generator.get_random_sample()
        target_string = ""
        prediction_string = ""
        #print(decoder_target_data.shape)
        for j in range(decoder_target_data.shape[0]):
            #print(decoder_target_data[j])
            decoded_word = decode(decoder_target_data[j])
            #print('WORD: ' + str(decoded_word))
            for x in trainer.data_generator.dict:
                #print(trainer.data_generator.dict[x])
                if trainer.data_generator.dict[x] == decoded_word:
                    #print('HELLO')
                    target_string = target_string + ' ' + x
                    prediction_string = prediction_string + ' ' + np.random.choice(keys, replace=True, p=probs)
        #print('Target: ' + target_string)
        #print('Prediciton: ' + prediction_string)
        acc = 1 - editdistance.eval(prediction_string, target_string) / np.maximum(len(prediction_string),
                                                                                   len(target_string))
        acc_total += acc
        if not i % 1000:
            print(str(i) + '/' + str(num))

    acc_total = acc_total / num
    print(acc_total)

# test_data_generation()
#print('########### Creating SIGNUM Data ###########')
#create_and_save_data_sig('/media/compute/homes/jkummert/data', 'signum_word')
#get_dumb_model_acc()

# Test pretrained model
#model = vgg_19_model.VGG_19('/home/johannes/Downloads/vgg19_weights.h5')
#model = K.applications.VGG19(weights='imagenet')
#K.utils.plot_model(model, to_file="model_vgg_19_full.png", show_shapes=True)

do_train = False
do_eval = True
if do_train:
    print('########### START CONV TRAINING WITH DATA_NAME:%s NUM_EPOCHS:%d LATENT_DIM:%d ###########' % (sys.argv[3], int(sys.argv[5]), int(sys.argv[7])))
    trainer, model = train_sig(model_path=sys.argv[1], data_location=sys.argv[2], data_name=sys.argv[3],
                           batch_size=int(sys.argv[4]), num_epochs=int(sys.argv[5]),
                           save_interval=int(sys.argv[6]), latent_dim=int(sys.argv[7]), arch=sys.argv[8], with_op=(sys.argv[9] == "True"))
elif do_eval:
    model_path = sys.argv[1]
    models = os.listdir(model_path)
    #models = ["net_2018-09-19_13:59:48.482086_stdLSTM_latent_dim_128.epoch5000"]
    for m in models:
        if m.endswith('.txt'):
            continue
        print(m)
        if not m.startswith("."):
            if os.path.isfile(model_path + m + '.txt'):
                continue
            orig_stdout = sys.stdout
            f = open(model_path + m + '.txt', 'w')
            sys.stdout = f
            loaded_model = model_path + m
            if '_256' in m:
                latent_dim = 256
            elif '_512' in m:
                latent_dim = 512
            elif '_1024' in m:
                latent_dim = 1024
            elif '_2048' in m:
                latent_dim = 2048
            elif '_64' in m:
                latent_dim = 64
            elif '_128' in m:
                latent_dim = 128
            if 'stdLSTM' in m:
                arch = "std"
            elif 'convLSTM' in m:
                arch = "std_conv"
            elif 'deepConvLSTM' in m:
                arch = "deep_conv"
            elif 'convMergeLSTM' in m:
                arch = "std_conv_merge"
            elif 'deepConvMergeLSTM' in m:
                arch = "deep_conv_merge"
            elif 'vgg_19_retrain_False' in m:
                arch = "vgg_19_retrain_False"
            elif 'vgg_19_retrain_True' in m:
                arch = "vgg_19_retrain_True"
            with_op = False
            if 'True' in m:
                with_op = True
            else:
                with_op = False
            print('########### EVALUATING MODEL %s WITH ARCH: %s, DIM: %d, OP: %s ###########' % (m,arch,latent_dim,with_op))
            acc = evaluate_image_model(loaded_model, arch=arch, op=with_op, latent_dim=latent_dim, num=250)
            print('\n########### MODEL ACCURACY: %f ###########' % acc)


# print('########### EVALUATE MODEL ###########')
# images = True
# if images:
#     ImageFile.LOAD_TRUNCATED_IMAGES = True
#     acc = 0
#     for i in range(100):
#         encoder_input_data, decoder_input_data, decoder_target_data = trainer.data_generator.get_random_image_sample()
#         acc += evaluate(model, trainer.data_generator.dict, int(sys.argv[7]),
#                    np.expand_dims(encoder_input_data, 0), np.expand_dims(decoder_target_data, 0), load_model=False)
#     acc = acc / 100
# else:
#     acc = evaluate(model, trainer.data_generator.dict, int(sys.argv[7]),
#                    trainer.data_generator.encoder_input_test, trainer.data_generator.decoder_output_test,
#                    load_model=False)
# print('\n########### MODEL ACCURACY: %f ###########' % acc)