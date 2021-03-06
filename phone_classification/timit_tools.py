from kaldi_tools import *
import tf_tools
import kaldi_io


def read_timit_labels(filename,fs=16000,win=0.025,inc=0.01):
    """
        read .PHN files from TIMIT dataset and return phone start/end labels
        that match the features.
        """
    s = [] # start of phone
    e = [] # end of phone
    l = [] # phone string
    [[s.append(int(i.split(' ')[0])/int(inc*fs)), # start sample / size of shift in samples
      e.append((int(i.split(' ')[1]) - int(win*fs))/int(inc*fs)), # end sample - window length / size of shift in samples
      l.append(i.strip().split(' ')[2])] for i in open(filename)]
    return s,e,l


def read_phn_scp(filename):
    """
        Take the file phn.scp as input and return file dictionary. 
        file dictionary contains the path to TIMIT transcription for 
        each utterance. 
        The input, phn.scp, must be generated by the user.
        """
    filedict = {}
    for i in open(filename):
        filedict[i.split(' ')[0]] = read_timit_labels(i.strip().split(' ')[1])
    return filedict


def phone_to_int(phone_list):
    """
        create a mapping from phone labels to integers.
        returns a dictionary phone_map = {'string':int}
        """
    phone_list.sort()
    phone_map = {}
    phone_set = set(phone_list)
    k = 0
    for i in phone_set:
        if not (i in phone_map):
            phone_map[i] = k
            k+=1
    return phone_map


def gen_data(root):
    """
        generate data from TIMIT directory. 
        The data generated here is designed to match tensorflow 
        I/O. Uses the function standard_array from tf_tools to 
        generate output.
        Input: location of TIMIT root directory, 'string'.
        
        Outputs:
            x_trn,x_tst: MFCC features for train and test
            y_trn,y_tst: integer labels for train and test
        """
    trn = root + '/trn/'
    tst = root + '/tst/'
    data = {}
    labels = {}

    for dataset in [trn,tst]:
        feat_reader = kaldi_io.SequentialBaseFloatMatrixReader('ark:%s/feat.ark'%(dataset))
        data[dataset] = read_ark(feat_reader)
        labels[dataset] = read_phn_scp('%s/phn.scp'%(dataset))
    
    # Use train labels to generate Map from phone strings to ints
    all_phones = []
    for i in labels[trn]:
        all_phones += labels[trn][i][2]

    phone_map = phone_to_int(all_phones)
    

    x_trn,y_trn = tf_tools.standard_array(data[trn], labels[trn], phone_map)
    x_tst,y_tst = tf_tools.standard_array(data[tst], labels[tst], phone_map)

    return x_trn,y_trn,x_tst,y_tst
