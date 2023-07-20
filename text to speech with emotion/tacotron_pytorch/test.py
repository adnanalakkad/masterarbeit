
import warnings
import librosa.display
from synthesis import tts as _tts
from tacotron_pytorch import Tacotron
from text import text_to_sequence, symbols
import soundfile as sf
from datasets.emovdb import vocab, get_test_data
from utils import get_last_checkpoint_file_name, load_checkpoint_test, load_checkpoint
from audio import save_to_wav, spectrogram2wav
from ssrn import SSRN
from text2mel import Text2Mel
import torch
import os
import sys
from matplotlib import widgets
from matplotlib.pyplot import imshow, xlabel, ylabel
import numpy as np
sys.path.append('pytorch-dc-tts/')
sys.path.append('pytorch-dc-tts/models')
sys.path.append("tacotron_pytorch/")
sys.path.append("tacotron_pytorch/lib/tacotron")

# For the DC-TTS


# For the Tacotron
# from util import audio


# For Audio/Display purposes

warnings.filterwarnings('ignore')

torch.set_grad_enabled(False)
text2mel = Text2Mel(vocab).eval()

ssrn = SSRN().eval()
load_checkpoint('trained_models/ssrn.pth', ssrn, None)

model = Tacotron(n_vocab=len(symbols),
                 embedding_dim=256,
                 mel_dim=80,
                 linear_dim=1025,
                 r=5,
                 padding_idx=None,
                 use_memory_mask=False,
                 )


def visualize(alignment, spectrogram, Emotion):
    label_fontsize = 16
    # tb = widgets.TabBar(['Alignment', 'Spectrogram'], location='top')
    #  with tb.output_to('Alignment'):
    #  imshow(alignment.T, aspect="auto", origin="lower", interpolation=None)
    #  xlabel("Decoder timestamp", fontsize=label_fontsize)
    #  ylabel("Encoder timestamp", fontsize=label_fontsize)
    # with tb.output_to('Spectrogram'):
    if Emotion == 'Disgust' or Emotion == 'Amused' or Emotion == 'Sleepiness':
        librosa.display.specshow(
            spectrogram.T, sr=fs, hop_length=hop_length, x_axis="time", y_axis="linear")
    else:
        librosa.display.specshow(
            spectrogram, sr=fs, hop_length=hop_length, x_axis="time", y_axis="linear")

        xlabel("Time", fontsize=label_fontsize)
        ylabel("Hz", fontsize=label_fontsize)


def tts_dctts(text2mel, ssrn, text):
    sentences = [text]

    max_N = len(text)
    L = torch.from_numpy(get_test_data(sentences, max_N)).long()
    zeros = torch.from_numpy(np.zeros((1, 80, 1), np.float32))
    Y = zeros
    A = None

    for t in range(210):
        _, Y_t, A = text2mel(L, Y, monotonic_attention=True)
        Y = torch.cat((zeros, Y_t), -1)
        _, attention = torch.max(A[0, :, -1], 0)
        attention = attention.item()
        if L[0, attention] == vocab.index('E'):  # EOS
            break

    _, Z = ssrn(Y)
    Y = Y.cpu().detach().numpy()
    A = A.cpu().detach().numpy()
    Z = Z.cpu().detach().numpy()

    return spectrogram2wav(Z[0, :, :].T), A[0, :, :], Y[0, :, :]


def tts_tacotron(model, text):
    waveform, alignment, spectrogram = _tts(model, text)
    return waveform, alignment, spectrogram


fs = 20000  # 20000


def present(waveform, Emotion, figures=False):

    if figures != False:
        visualize(figures[0], figures[1], Emotion)
    sf.write("Test.wav", waveform, fs)
    # IPython.display.display(Audio(waveform, rate=fs))


hop_length = 250
model.decoder.max_decoder_steps = 200

# @title Select the emotion and type the text

Emotion = "Amused"  # @param [ "Angry",  "Amused"]
Text = 'Today is a beautiful day with clear skies.'  # @param {type:"string"}

wav, align, mel = None, None, None

if Emotion == "Neutral":
    load_checkpoint('trained_models/'+Emotion.lower() +
                    '_dctts.pth', text2mel, None)
    wav, align, mel = tts_dctts(text2mel, ssrn, Text)
elif Emotion == "Angry" or Emotion == "Disgust":
    load_checkpoint_test('trained_models/'+Emotion.lower() +
                         '_dctts.pth', text2mel, None)
    wav, align, mel = tts_dctts(text2mel, ssrn, Text)
    # wav = wav.T
elif Emotion == "Amused" or Emotion == "Sleepiness":
    checkpoint = torch.load('trained_models/'+Emotion.lower() +
                            '_tacotron.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    wav, align, mel = tts_tacotron(model, Text)

present(wav, Emotion, (align, mel))
