import os
import shutil
import string
import re 


PATH = "./emovdb/"

emotions = {}

speakers = os.listdir(PATH)

for speaker in speakers:
    speaker_path = PATH+speaker+"/"
    print(speaker_path)
    speaker_emotions = os.listdir(speaker_path)
    for emotion in speaker_emotions:
        if emotion not in emotions:
            emotions[emotion] = []
        emotions[emotion].append(speaker_path+emotion)

PROCESSED_DIR = './processed_emovdb_disgust'
if os.path.isdir(PROCESSED_DIR) == False:
    os.mkdir(PROCESSED_DIR)
    os.mkdir(PROCESSED_DIR+'/wavs')

TRANSCRIPT_PATH = './cmuarctic.data'


def remove_punct(string): 
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, "") 
                                          
    return string.lower()


def process_transcript(transcript):

    with open(transcript, 'r') as f:
        lines = f.readlines()

    label_to_transcript = {}
    
    for line in lines:
        line = line.split('"')
        sent = line[1]
        label = line[0].rstrip().split('_')[-1]
        if label[0] == 'b':
            continue
        label = label[1:]
        label_to_transcript[label] = remove_punct(sent)

    return label_to_transcript

label_to_transcript = process_transcript(TRANSCRIPT_PATH)

list_transcript_files = []
for emotion in emotions.keys():
    # Remove this to preprocess on all emotions
    if emotion != 'disgust':
        continue
    emotion_paths = emotions[emotion]
    count_emotion = 0
    for path in emotion_paths:
        audios = os.listdir(path)
        speaker = path.split('/')[-2]
        for audio in audios:
            #print('-------------------------')
            #print(audio)
            label = audio.split('_')[-1].split('.')[0]
            #print(label)
            label_name = emotion+"_"+str(count_emotion)+"_"+speaker+"_"+label
            #print(label_name)
            count_emotion += 1
            try:
                list_transcript_files.append(label_name+"|"+label_to_transcript[label])
                shutil.copyfile(path+'/'+audio,PROCESSED_DIR+'/wavs/'+label_name+".wav")
            except:
                print('Failed for file - ', path+'/'+audio)
    print('Done emotion - ', emotion)
    print('Count - ', count_emotion)

with open(PROCESSED_DIR+"/transcript.csv","w+") as f:
    f.write("\n".join(list_transcript_files))

print('File Writing Done')





