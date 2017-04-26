import kaldi_io

def read_ark(feat_reader):
    hasfeat = True
    featdict = {}
    while (hasfeat):
        uttid = ''
        try:
            (uttid,feat) = next(feat_reader)
        except:
            hasfeat = False
        if (uttid!=''):
            featdict[uttid] = feat
    return featdict
