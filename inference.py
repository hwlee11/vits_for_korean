#%matplotlib inline
#import matplotlib.pyplot as plt
#import IPython.display as ipd
import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.korean.symbols import symbols
from text.korean import text_to_sequence

from scipy.io.wavfile import write
import numpy as np


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("./configs/kss_base.json")
sr = hps['data']['sampling_rate']

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model)#.cuda()
_ = net_g.eval()
#_ = utils.load_checkpoint("logs/kss_base/G_315000.pth", net_g, None)
_ = utils.load_checkpoint("logs/kss_nofp3/G_9800.pth", net_g, None)
#'logs/kss_base/G_44000.pth'


#stn_tst = get_text("뽁구는 탈모다.", hps)
stn_tst = get_text("알로에쿤!", hps)
with torch.no_grad():
    x_tst = stn_tst.unsqueeze(0)
    #x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])#.cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    print(audio,len(audio))
    write('test.wav', sr, audio.astype(np.float32))
#ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))




