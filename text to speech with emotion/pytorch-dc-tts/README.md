# Pytorch Implementation of DC-TTS for Emotional TTS

This fork is modified to work for transfer learning for low-resource emotional TTS, as described [here](https://github.com/Emotional-Text-to-Speech/dl-for-emo-tts#approach-dctts-models). 

## Training

1. Install the dependencies using ```pip install -r requirements.txt```
1. Preprocess the [EmoV-DB](https://github.com/Emotional-Text-to-Speech/dl-for-emo-tts#datasets) dataset using ```process_emovdb.py```
1. Change the **```logdir```** argument in ```hyperparams.py```. Other parameters can be edits optionally. __DO NOT__ edit [these](https://github.com/Emotional-Text-to-Speech/pytorch-dc-tts/blob/master/hyperparams.py#L13-L31) hyperparameters.
1. Add the path to the pre-trained **Text2Mel** model in the **logdir**
1. Comment [this line](https://github.com/Emotional-Text-to-Speech/pytorch-dc-tts/blob/master/datasets/emovdb.py#L56) if you are *not* running the ```train-text2mel.py``` file for the first time.
1. Run the training script like - ```python train-text2mel.py --dataset=emovdb```

## Synthesis

1. Write the sentences that you want to generate [here](https://github.com/Emotional-Text-to-Speech/pytorch-dc-tts/blob/master/synthesize.py#L50-L57)
1. Add the checkpoint for the fine-tuned Text2Mel model [in place of this line](https://github.com/Emotional-Text-to-Speech/pytorch-dc-tts/blob/master/synthesize.py#L81)
1. Edit the [paths for the output](https://github.com/Emotional-Text-to-Speech/pytorch-dc-tts/blob/master/synthesize.py#L123-L126).
1. Run the synthesis script like - ```python synthesize.py -- dataset=emovdb```



----
----

Readme of the original repository


PyTorch implementation of
[Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969)
based partially on the following projects:
* https://github.com/Kyubyong/dc_tts (audio pre processing)
* https://github.com/r9y9/deepvoice3_pytorch (data loader sampler)

## Online Text-To-Speech Demo
The following notebooks are executable on [https://colab.research.google.com ](https://colab.research.google.com):
* [Mongolian Male Voice TTS Demo](https://colab.research.google.com/github/tugstugi/pytorch-dc-tts/blob/master/notebooks/MongolianTTS.ipynb)
* [English Female Voice TTS Demo (LJ-Speech)](https://colab.research.google.com/github/tugstugi/pytorch-dc-tts/blob/master/notebooks/EnglishTTS.ipynb)

For audio samples and pretrained models, visit the above notebook links.

## Training/Synthesizing English Text-To-Speech
The English TTS uses the [LJ-Speech](https://keithito.com/LJ-Speech-Dataset/) dataset.
1. Download the dataset: `python dl_and_preprop_dataset.py --dataset=ljspeech`
2. Train the Text2Mel model: `python train-text2mel.py --dataset=ljspeech`
3. Train the SSRN model: `python train-ssrn.py --dataset=ljspeech`
4. Synthesize sentences: `python synthesize.py --dataset=ljspeech`
   * The WAV files are saved in the `samples` folder.

## Training/Synthesizing Mongolian Text-To-Speech
The Mongolian text-to-speech uses 5 hours audio from the [Mongolian Bible](https://www.bible.com/mn/versions/1590-2013-ariun-bibli-2013).
1. Download the dataset: `python dl_and_preprop_dataset.py --dataset=mbspeech`
2. Train the Text2Mel model: `python train-text2mel.py --dataset=mbspeech`
3. Train the SSRN model: `python train-ssrn.py --dataset=mbspeech`
4. Synthesize sentences: `python synthesize.py --dataset=mbspeech`
   * The WAV files are saved in the `samples` folder.
