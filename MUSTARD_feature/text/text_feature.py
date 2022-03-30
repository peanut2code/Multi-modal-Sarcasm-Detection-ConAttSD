import sys
import json
import random
import re
random.seed(2021)
import torch
from transformers import BertModel, BertTokenizer
import en_core_web_sm
import spacy
import unidecode
from bs4 import BeautifulSoup
import pandas as pd
import pickle

nlp = spacy.load('en_core_web_sm')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

if __name__ == "__main__":
    data1 = pd.read_excel('MUSTARD/MUSTARD.xlsx', header=0)
    data = data1.loc[:, ['KEY','SPEAKER', 'SENTENCE', 'SHOW', 'SARCASM','SENTIMENT_IMPLICIT','SENTIMENT_EXPLICIT','EMOTION_IMPLICIT','EMOTION_EXPLICIT','MINTIME','MAXTIME','ALLTIME']]

    keys=data['KEY']
    sentences=data['SENTENCE']
    features = {}
    num=0
    for i in range(len(sentences)):
        if pd.notnull(keys[i]):
            print(keys[i])
            num+=1
            text_feature = {}
            sentence_list = []
            text = sentences[i]
            encoded = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
            outputs = model(encoded)
            vector = outputs[1]
            text_feature['text']=vector
            visual_name_noext = keys[i]
            features[visual_name_noext] = text_feature

    with open('Feature/text_features.pkl', 'wb') as f:
        pickle.dump(features, f)