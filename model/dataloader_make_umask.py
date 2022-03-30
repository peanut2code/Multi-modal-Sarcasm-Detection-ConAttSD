import random
import os
import numpy as np
import torch
def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
seed_torch(20210412)

class MUSTARDDataset(Dataset):

    def __init__(self, path, train=True):
        with open(path, 'rb') as file:
            data=pickle.load(file, encoding='latin1')
        self.videoIDs = data[0]
        self.videoSpeakers = data[1]
        self.sarcasmslabel = data[2]
        self.sentiment_implicit=data[3]
        self.sentiment_explicit=data[4]
        self.videoText = data[7]
        self.videoAudio = data[8]
        self.videoVisual = data[9]
        self.videoSentence = data[10]
        self.trainVid = sorted(data[11])
        self.testVid = sorted(data[12])
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, index):

        vid = self.keys[index]
        #true
        umask=[]
        labellen=len(self.sarcasmslabel[vid])
        for i in range (labellen):
            if i!= labellen-1:
                umask.append(0)
            else:
                umask.append(1)
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoVisual[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor(self.videoSpeakers[vid]), \
               torch.FloatTensor(umask), \
               torch.LongTensor(self.sarcasmslabel[vid]), \
               torch.LongTensor(self.sentiment_implicit[vid]), \
               torch.LongTensor(self.sentiment_explicit[vid]), \
               vid



    def __len__(self):
        return self.len

    def collate_fn(self, data):

        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<8 else dat[i].tolist() for i in dat]