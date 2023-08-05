from django.shortcuts import render
from django.http import JsonResponse

import random
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
from keras import Input, Model
from keras.layers import Embedding, LSTM, Flatten,Input,Dense,GlobalMaxPool1D
from nltk.stem import WordNetLemmatizer
from keras.models import Model
nltk.download('wordnet')
import string
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
# import gradio




def chatbot_response(user_input):
    
    
    # user= request.POST.get("user")
    with open ('ren/intents.json') as content:
        data1=json.load(content)
    tag=[]
    inputs=[]
    responses={}
    for intent in data1['intents']:
        responses[intent['tag']]=intent['responses']
        for lines in intent['patterns']:
            inputs.append(lines)
            tag.append(intent['tag'])

    data= pd.DataFrame({'input':inputs,'tags':tag})

    data['input']= data['input'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
    data['input']= data['input'].apply(lambda wrd:''.join(wrd))
    # print(data)

    tokenizer= Tokenizer(num_words=2000)
    tokenizer.fit_on_texts(data['input'])
    train= tokenizer.texts_to_sequences(data['input'])
    X_train= pad_sequences(train)

    le= LabelEncoder()
    Y_train = le.fit_transform(data['tags'])

    input_shape= X_train.shape[-1]
    # print(input_shape)

    vocabulary=  len(tokenizer.word_index)
    # print(vocabulary)
    output_length= le.classes_.shape[0]
    # print(output_length)

    # creating a model
    i=Input(shape=(input_shape,))
    x= Embedding(vocabulary+1,10)(i)
    x= LSTM(10,return_sequences=True)(x)
    x= Flatten()(x)
    x=Dense(output_length,activation='softmax')(x)
    model= Model(i,x)

    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=["accuracy"])

    train=model.fit(X_train,Y_train,epochs=200)
    # print(train)

    # chatting

    while True:
  
        
        texts_p=[]
        # user= request.POST.get("user")
        prediction_output= user_input

        # removing punctuation
        prediction_output=[letters.lower() for letters in prediction_output if letters not in string.punctuation]
        prediction_output=''.join(prediction_output)
        texts_p.append(prediction_output)

        # tokenizing and padding
        prediction_output= tokenizer.texts_to_sequences(texts_p)
        prediction_output= np.array(prediction_output).reshape(-1)
        prediction_output= pad_sequences([prediction_output],input_shape)

        # getting output from model
        output= model.predict(prediction_output)
        output= output.argmax()

        # finding the right tag and predicting
        response_tag= le.inverse_transform([output])[0]
        g= random.choices(responses[response_tag])
        # if responses=='goodbye':
        #     break


        return g

def chatbot(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        bot_response = chatbot_response(user_input)
        return JsonResponse({'response': bot_response})

    return render(request, 'chat.html')
