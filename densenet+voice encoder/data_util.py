import torch
import collections
import os
import soundfile as sf
import librosa
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import pickle

ASVFile = collections.namedtuple('ASVFile',
    ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])

class ASVDataset(Dataset):
    def __init__(self, data_type=None):
        self.data_type = data_type
        if self.data_type=='dev':
            self.cache_fname = 'cache_speech2face_dev.npy'
            self.protocols_fname = 'ASVspoof2019.LA.cm.dev.trl.txt'
            self.files_dir = './data/temp/dev'
        if self.data_type=='train':
            self.cache_fname = 'cache_speech2face_train.npy'
            self.protocols_fname = 'ASVspoof2019.LA.cm.train.trn.txt'
            self.files_dir = './data/temp/train'
        if self.data_type=='eval':
            self.cache_fname = 'cache_speech2face_eval.npy'
            self.protocols_fname = 'ASVspoof2019.LA.cm.eval.trl.txt'
            self.files_dir = './data/temp/eval'
        self.sysid_dict = {
            '-': 0,  # bonafide speech
            'A01': 1, # Wavenet vocoder
            'A02': 2, # Conventional vocoder WORLD
            'A03': 3, # Conventional vocoder MERLIN
            'A04': 4, # Unit selection system MaryTTS
            'A05': 5, # Voice conversion using neural networks
            'A06': 6, # transform function-based voice conversion
            'A07': 7,
            'A08': 8,
            'A09': 9,
            'A10': 10,
            'A11': 11,
            'A12': 12,
            'A13': 13,
            'A14': 14,
            'A15': 15,
            'A16': 16,
            'A17': 17,
            'A18': 18,
            'A19': 19,
            # For PA:
            'AA':7,
            'AB':8,
            'AC':9,
            'BA':10,
            'BB':11,
            'BC':12,
            'CA':13,
            'CB':14,
            'CC': 15
        }

        self.sysid_dict_inv = {v: k for k, v in self.sysid_dict.items()}

        if os.path.exists(self.cache_fname):
            self.data_x, self.data_y, self.data_sysid, self.files_meta = torch.load(self.cache_fname)
            print('Dataset loaded from cache ', self.cache_fname)
        else:
            self.files_meta = self.parse_protocols_file(self.protocols_fname)
            data = list(map(self.read_file, self.files_meta))
            self.data_x, self.data_y, self.data_sysid = map(list, zip(*data))
            torch.save((self.data_x, self.data_y, self.data_sysid, self.files_meta), self.cache_fname)
            print('Dataset saved to cache ', self.cache_fname)
        self.lenth = len(self.data_x)
    def __getitem__(self, index):
        return self.data_x[index],self.data_y[index], self.files_meta[index]

    def __len__(self):
        return self.lenth


    def read_file(self, meta):
        f = open(meta.path, 'rb')
        data_x = pickle.load(f)
        data_x = data_x.reshape((64, 64))
        data_y = meta.key
        return data_x, float(data_y), meta.sys_id

    def _parse_line(self, line):
        tokens = line.strip().split(' ')
        print(len(tokens))
        return ASVFile(speaker_id=tokens[0],
            file_name=tokens[1],
            path=os.path.join(self.files_dir, tokens[1] + '.pkl'),
            sys_id=self.sysid_dict[tokens[3]],
            key=int(tokens[4] == 'bonafide'))

    def parse_protocols_file(self, protocols_fname):
        lines = open(protocols_fname).readlines()
        files_meta = map(self._parse_line, lines)
        return list(files_meta)