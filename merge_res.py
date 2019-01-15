import numpy as np
import pickle
import test_data
import input_data
#0.1
#0.3
#0.6
#0.8
ratio = 0.1
def pickle_dump(obj, fn ):
    with open(fn,'wb') as f:
        pickle.dump(obj, f)

def pickle_load(fn):
    obj = None
    with open(fn,'rb') as f:
        obj = pickle.load(f)
    return obj
import os
def merge(file_dir, files, names_file):
    probs = []
    for file in files:
        if os.path.exists(file_dir+file):
            with open(file_dir+file, 'rb') as fin:
                try:
                    prob = pickle.load(fin)
                    probs.append(prob)
                except:
                    print('error', file)
        else:
            print(file,'not found')
    prob_m = np.zeros( probs[0].shape, probs[0].dtype)
    for prob in probs:
        prob_m += prob/len(probs)
    indices = np.argmax(prob_m, axis=1)
    res_files, words_list = pickle_load(names_file)
    res_words = [words_list[indice] for indice in indices]
    with open('res.csv', 'w') as fout:
        fout.write('fname,label\n')
        if len(res_words)> len(res_files):
            res_files = res_files[:len(res_words)]
        for f, word in zip(res_files, res_words):
            real_word = word
            if word == test_data.SILENCE_LABEL:
                real_word = 'silence'
            elif word == test_data.UNKNOWN_WORD_LABEL:
                real_word = 'unknown'
            if real_word == test_data.SILENCE_LABEL or real_word == test_data.UNKNOWN_WORD_LABEL:
                real_word = 'silence'
            fout.write('{},{}\n'.format(f,real_word))
    print('done.')

def merge_and_fix(file_dir, files, names_file):
    probs = []
    for file in files:
        with open(file_dir+file, 'rb') as fin:
            prob = pickle.load(fin)
            probs.append(prob)
    prob_m = np.zeros( probs[0].shape, probs[0].dtype)
    for prob in probs:
        prob_m += prob/len(probs)
    indices0 = np.argmax(prob_m, axis=1)
    print('num unk', np.sum(indices0 == 1))
    for i in range(prob_m.shape[0]):
        for j in range(prob_m.shape[1]):
            if j==1:
                prob_m[i,j] = prob_m[i,j]*7.71
            else:
                prob_m[i, j] = prob_m[i, j] / 2.59
    indices = np.argmax(prob_m, axis=1)
    print('num unk after fix', np.sum(indices == 1))
    res_files, words_list = pickle_load(names_file)
    res_words = [words_list[indice] for indice in indices]
    with open('res_fix.csv', 'w') as fout:
        fout.write('fname,label\n')
        if len(res_words)> len(res_files):
            res_files = res_files[:len(res_words)]
        for f, word in zip(res_files, res_words):
            real_word = word
            if word == test_data.SILENCE_LABEL:
                real_word = 'silence'
            elif word == test_data.UNKNOWN_WORD_LABEL:
                real_word = 'unknown'
            if real_word == test_data.SILENCE_LABEL or real_word == test_data.UNKNOWN_WORD_LABEL:
                real_word = 'silence'
            fout.write('{},{}\n'.format(f,real_word))
    print('done.')

def merge_and_fix_s(file_dir, files, names_file):
    probs = []
    for file in files:
        with open(file_dir+file, 'rb') as fin:
            prob = pickle.load(fin)
            probs.append(prob)
    prob_m = np.zeros( probs[0].shape, probs[0].dtype)
    for prob in probs:
        prob_m += prob/len(probs)
    sep_file = 'out/cnnblstm_s.pkl'
    #0: unk, 1:know
    prob_sep = pickle_load(sep_file)
    #TODO
    indices0 = np.argmax(prob_m, axis=1)
    print('num unk', np.sum(indices0 == 1))
    for i in range(prob_m.shape[0]):
        prob_known = np.sum(prob_m[i,:]) - prob_m[i,1]
        for j in range(prob_m.shape[1]):
            if j == 1:
                prob_m[i, j] = prob_sep[i, 0]
            else:
                prob_m[i, j] = prob_m[i, j]/(prob_known+1e-8) *prob_sep[i,1]
    indices = np.argmax(prob_m, axis=1)
    print('num unk after fix', np.sum(indices == 1))
    res_files, words_list = pickle_load(names_file)
    res_words = [words_list[indice] for indice in indices]
    with open('res_fix_s.csv', 'w') as fout:
        fout.write('fname,label\n')
        if len(res_words)> len(res_files):
            res_files = res_files[:len(res_words)]
        for f, word in zip(res_files, res_words):
            real_word = word
            if word == test_data.SILENCE_LABEL:
                real_word = 'silence'
            elif word == test_data.UNKNOWN_WORD_LABEL:
                real_word = 'unknown'
            if real_word == test_data.SILENCE_LABEL or real_word == test_data.UNKNOWN_WORD_LABEL:
                real_word = 'silence'
            fout.write('{},{}\n'.format(f,real_word))
    print('done.')

def fix_indices(prob_m, wanted_words, model_arc):
    all_wanted_words = 'yes,no,up,down,left,right,on,off,stop,go'.split(',')
    all_wanted_words = input_data.prepare_words_list( all_wanted_words )
    wanted_words_indices = []
    for word in wanted_words:
        for i, w in enumerate(all_wanted_words):
            if w == word:
                wanted_words_indices.append(i)
                break
    suffix = '_'.join([word for word in wanted_words])
    fn = 'out/{}_{}.pkl'.format(model_arc, suffix)
    probs = pickle_load(fn)
    #fix
    # ratio = 0.3
    print(wanted_words_indices)
    for i in range(prob_m.shape[0]):
        ww_prob_sum = 0.0
        for j in range(prob_m.shape[1]):
            if j in wanted_words_indices:
                ww_prob_sum = ww_prob_sum+prob_m[i,j]
        for j in range(prob_m.shape[1]):
            if j in wanted_words_indices:
                real_j = 0
                for ii, wwi in enumerate(wanted_words_indices):
                    if wwi == j:
                        real_j = ii
                        break
                prob_m[i,j] = prob_m[i,j]*(1-ratio)+probs[i,real_j]*ww_prob_sum*ratio
import os
def merge_and_fix_ss(file_dir, files, names_file):
    probs = []
    for file in files:
        if os.path.exists(file_dir + file):
            with open(file_dir+file, 'rb') as fin:
                try:
                    prob = pickle.load(fin)
                    probs.append(prob)
                except:
                    print('error', file)
        else:
            print(file, 'not find')

    prob_m = np.zeros( probs[0].shape, probs[0].dtype)
    for prob in probs:
        prob_m += prob/len(probs)
    sep_file = 'out/cnnblstm_s.pkl'
    #0: unk, 1:know
    # ratio = 0.3
    prob_sep = pickle_load(sep_file)
    indices0 = np.argmax(prob_m, axis=1)
    num_label = prob_m.shape[1]
    print('num unk', np.sum(indices0 ==1))
    for i in range(prob_m.shape[0]):
        prob_known_avg = prob_sep[i,1]/(num_label-1)
        for j in range(prob_m.shape[1]):
            if j == 1:
                prob_m[i, j] = (1-ratio)*prob_m[i,j]+ratio*prob_sep[i, 0]
            else:
                prob_m[i, j] = (1-ratio)*prob_m[i, j]+ratio *prob_known_avg
    indices = np.argmax(prob_m, axis=1)
    print('num unk after fix', np.sum(indices == 1))
    wanted_words_set = ['up,off', 'go,down,no', 'stop,go']
    model_arc = 'cnnblstm'
    for wanted_words_str in wanted_words_set:
        wanted_words = wanted_words_str.split(',')  #
        fix_indices(prob_m, wanted_words, model_arc)


    res_files, words_list = pickle_load(names_file)
    res_words = [words_list[indice] for indice in indices]
    with open('res_fix_ss.csv', 'w') as fout:
        fout.write('fname,label\n')
        if len(res_words)> len(res_files):
            res_files = res_files[:len(res_words)]
        for f, word in zip(res_files, res_words):
            real_word = word
            if word == test_data.SILENCE_LABEL:
                real_word = 'silence'
            elif word == test_data.UNKNOWN_WORD_LABEL:
                real_word = 'unknown'
            fout.write('{},{}\n'.format(f,real_word))
    print('done.')


if __name__ == '__main__':
    files = [ 'resnet.pkl', 'resnet101.pkl','resnetblstm.pkl',
              'xception.pkl',
              'cnnblstm.pkl', 'cnnblstmbnd.pkl','se_resnet.pkl',
              'lace2.pkl', 'lace.pkl','cnnblstm2.pkl',  'cnnblstm6.pkl',
              #extra data
              'cnnblstm_1.pkl',
              #normmfcc
              'lace_normmfcc.pkl','cnnblstm_normmfcc.pkl','resnet_normmfcc.pkl',
              'lace2_normmfcc.pkl',
              'resnetblstm_normmfcc.pkl','resnet101_normmfcc.pkl','xception_normmfcc.pkl',
              'cnnblstmbnd_normmfcc.pkl','cnnblstm2_normmfcc.pkl','cnnblstm6_normmfcc.pkl','se_resnet_normmfcc.pkl',
              'resnet_dmfcc.pkl','cnnblstm_dmfcc.pkl','resnet_dmfcc.pkl',
              'lace2_dmfcc.pkl','lace_dmfcc.pkl',
              'resnetblstm_dmfcc.pkl','resnet101_dmfcc.pkl','xception_dmfcc.pkl',
              'cnnblstmbnd_dmfcc.pkl', 'cnnblstm2_dmfcc.pkl', 'cnnblstm6_dmfcc.pkl','se_resnet_dmfcc.pkl',
              #filter bank
              'resnetblstm_fb.pkl','resnet_fb.pkl','resnet101_fb.pkl',
              'cnnblstmbnd_fb.pkl',
                'cnnblstm2_fb.pkl', 'cnnblstm6_fb.pkl', 'lace_fb.pkl', 'lace2_fb.pkl'
              ]#,'cnnblstm_2.pkl'
    #,
    #
    #

    names_file = 'out/names.pkl'
    file_dir = 'out/'
    merge(file_dir, files, names_file)
    # merge_and_fix(file_dir, files, names_file)
    # merge_and_fix_s(file_dir, files, names_file)
    merge_and_fix_ss(file_dir, files, names_file)