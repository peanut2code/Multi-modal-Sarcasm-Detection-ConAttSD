import pandas as pd
import os
import re
import openpyxl
from pydub import AudioSegment
import cv2

#extract video frame
if __name__ == "__main__":

    data1 = pd.read_excel('MUSTARD/MUSTARD.xlsx', header=0)
    data = data1.loc[:, ['KEY','SPEAKER', 'SENTENCE', 'SHOW', 'SARCASM','SENTIMENT_IMPLICIT','SENTIMENT_EXPLICIT','EMOTION_IMPLICIT','EMOTION_EXPLICIT','MINTIME','MAXTIME','ALLTIME']]

    keys=data['KEY']
    mintime=data['MINTIME']
    maxtime=data['MAXTIME']
    alltime=data['ALLTIME']

    context_list=[]
    dir = 'Video/context_video/'
    for filename in os.listdir(dir):
        context_list.append(filename)
    utterance_list=[]
    dir = 'Video/utterances_video/'
    for filename in os.listdir(dir):
        utterance_list.append(filename)
    for i in range(len(keys)):
        if pd.notnull(keys[i]):
            if 'utterance' in keys[i]:
                index = keys[i].find('_utterances')
                tempname = keys[i][0:index]
                for utfile in utterance_list:
                    if tempname+'.mp4' == utfile:
                        print(keys[i])
                        audiofile='Video/utterances_video/'+utfile

                        vc = cv2.VideoCapture(audiofile)
                        rate = vc.get(5)
                        fraNum = vc.get(7)

                        duration = (fraNum / rate)*1000
                        begin = round(mintime[i], 5)
                        end = round(maxtime[i], 5)
                        all = round(alltime[i], 5)
                        begintime = (begin / all) * duration
                        endtime = (end / all) * duration
                        midtime=(begintime+endtime)/2
                        vc.set(cv2.CAP_PROP_POS_MSEC, midtime)  #
                        rval, frame = vc.read()
                        cv2.imwrite('picture/'+keys[i]+'.jpg', frame)
            else:
                index = keys[i].find('##')
                tempname = keys[i][0:index]
                for confile in context_list:
                    if tempname + '_c.mp4' == confile:
                        audiofile = 'Video/context_video/' + confile
                        print(keys[i])
                        # 抽取帧
                        vc = cv2.VideoCapture(audiofile)
                        rate = vc.get(5)
                        fraNum = vc.get(7)
                        duration = (fraNum / rate) * 1000
                        begin = round(mintime[i], 5)
                        end = round(maxtime[i], 5)
                        all = round(alltime[i], 5)
                        begintime = (begin / all) * duration
                        endtime = (end / all) * duration
                        midtime = (begintime + endtime) / 2
                        vc.set(cv2.CAP_PROP_POS_MSEC, midtime)
                        rval, frame = vc.read()
                        cv2.imwrite('picture/'+keys[i]+'.jpg', frame)