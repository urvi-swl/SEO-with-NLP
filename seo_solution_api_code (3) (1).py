##############IMPORTING PACKAGES##################
import json
import pickle
import pandas as pd
# import library tkinter
import tkinter as tk
from tkinter import *
import re
from collections import OrderedDict
#import cache
import time
import numpy as np
from os import listdir
from os.path import isfile, join
####################################################################
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
###################################################################
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words=set(stopwords.words('english'))
w_tokenizer=nltk.tokenize.WhitespaceTokenizer()
lemmatizer=nltk.stem.WordNetLemmatizer()
from flask import Flask,  request, json
from flask import jsonify
from flask import Flask
from flask_cors import CORS
import arpa 
#from bert5 import *
#from ensemble import *
from symspellpy import SymSpell,Verbosity
############################BERT ###################3
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import bert
from tqdm import tqdm
import numpy as np
#initialising the app
app = Flask(__name__, static_folder='/var/www/flaskapp/gemnlp/build', static_url_path='/')
CORS(app)
print("TensorFlow Version:",tf.__version__)
print("Hub version: ",hub.__version__)
from sklearn.preprocessing import LabelEncoder
import time
with open("C:\\Users\harsh\\Downloads\\seo\\gem_local_ver2\\cat_internal.txt")as f:
    map_cat=f.read().splitlines()
encoder = LabelEncoder()
encoder.fit(map_cat)
encoded_Y = encoder.transform(map_cat)
path_model='C:/Users/harsh/Downloads/seo/bert_model_ver2_internal_tf_2.6.2/bert_model_ver2_internal_tf_2.6.2/my_model_phase4_level1'
model_bert = tf.keras.models.load_model(path_model)
path='C:\\Users\\harsh\\Downloads\\seo\\gem_local_ver2\\'
max_seq_length = 40
MAX_SEQ_LEN = 40
MAX_LEN = 40
bs = 32 # Your choice here.
FullTokenizer=bert.bert_tokenization.FullTokenizer
#path_bert='C:\\Users\\anshu.chaudhary\\OneDrive - NEC\\Desktop\\gem\\phase4_with_preprocessing\\model\\uncased_L-12_H-768_A-12\\'
path_bert='C:/Users/harsh/Downloads/model/model/uncased_L-12_H-768_A-12/'
tokenizer = FullTokenizer(vocab_file=path_bert+ "vocab.txt")
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,name="segment_ids")
def get_masks(tokens, max_seq_length):
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens,)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def create_single_input(sentence,MAX_LEN):
  
    stokens = tokenizer.tokenize(sentence)

    stokens = stokens[:MAX_LEN]

    #stokens = ["[CLS]"] + stokens + ["[SEP]"]

    ids = get_ids(stokens, tokenizer, MAX_SEQ_LEN)
    masks = get_masks(stokens, MAX_SEQ_LEN)
    segments = get_segments(stokens, MAX_SEQ_LEN)

    return ids,masks,segments

def create_input_array(sentences):

    input_ids, input_masks, input_segments = [], [], []

    for sentence in tqdm(sentences,position=0, leave=True):
        #print(sentence)
        ids,masks,segments=create_single_input(sentence,MAX_SEQ_LEN-2)

        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]
#FullTokenizer=bert.bert_tokenization.FullTokenizer
path_gen='C:\\Users\\harsh\\Downloads\\seo\\gem_local_ver2\\'
with open(path_gen+"stopwords.txt","r")as f:
    r=f.read().splitlines()
    print(len(r))
###############################################################################################################
def load_svm():
    ######################SVM MODEL FOR SEARCH ENGINE CATEGORY####################################
    with open(path+"gem_internal_ver2_SVM.pkl", 'rb') as file:
        model_svm = pickle.load(file)

    print("Model has been loaded succesfully")

    print("Model has "+str(len(model_svm.classes_))+" no of class")
    return model_svm
model_svm=load_svm()    
##############################################################################################
def level1_preprocess_bert(orig_str):
    
    ########REMOVE DIGITS##########3
    mystr_digits_removed=''.join('' if c.isdigit() else c for c in orig_str)
    #print("digits removed=====",mystr_digits_removed)

    #############REMOVE SPECIAL CHARS###############


    #mystr_special_removed= re.sub('[^a-zA-Z0-9 \n\.]', '', my_str)
    mystr_special_removed= re.sub('[^a-zA-Z0-9 \n\.]', ' ',mystr_digits_removed)
    #print("special chars removed=======",mystr_special_removed)    



    mystr_special_removed=mystr_special_removed.replace('.',' ')
    mystr_special_removed = mystr_special_removed.replace('  ', ' ') 
    mystr_special_removed=mystr_special_removed.strip()
    
    #print("dot removed=====",mystr_special_removed)  

    ##############REMOVE STOPWORDS##############
       

  
    

    
    word_tokens = word_tokenize(mystr_special_removed)

    filtered_sentence = [w for w in word_tokens if not w in r] 
    #print(filtered_sentence)
    filter_sentence=(' '.join(filtered_sentence))


    ###################DO LEMMATIZATION##############################
    t=lemmatize_text(filter_sentence)
    
    return t
def model_amazon(m,txt):
    
    #y_train = gem_df['Category Name']
    test_string=level1_preprocess_bert(txt)
    predd=[]
    list1=[test_string]
    for i in list1:
        print("PREPROCESSD STRING=====",i)
        X_train=[i]
        inputs=create_input_array(X_train)
        encoder = LabelEncoder()
        encoder.fit(map_cat)
        encoded_Y = encoder.transform(map_cat)
        print('\n')
        pred = m.predict(inputs)
        #print(np.argsort(-pred))
        #print("print1",(-np.sort(-pred)))
        q=np.argsort(-pred)
        b2=[encoder.inverse_transform([q[0][0]]),encoder.inverse_transform([q[0][1]])]
        print(pred[0][q[0][0]])
        b=list(encoder.inverse_transform(np.apply_along_axis(np.argmax,1,pred)))
        #print(np.apply_along_axis(np.argmax,1,pred))
    #print(i,b)
        #pp=np.apply_along_axis(np.max,1,pred)
    #print("String tested==== \'{title}\' and CLASS PREDICTED ===={catt}".format(title = i,catt = b))
        print("String tested==== \'{title}\' and CLASS PREDICTED 1 ===={catt1}---{prob1} AND CLASS PREDICTED 2===={catt2}==={prob2}".format(title = i,catt1 = b2[0],prob1=pred[0][q[0][0]],catt2=b2[1],prob2=pred[0][q[0][1]]))
        predd.append(b)
    #return predd[0][0]
    
    return b2,[pred[0][q[0][0]],pred[0][q[0][1]]]
###############--------SETTING UNIVERSAL COMMON PATH----------------###########################

#path2 = '/home/anshu/anshu/gem_phase4/phase4_with_preprocess_v1/'

##################################################################
def load_svm():
    ######################SVM MODEL FOR SEARCH ENGINE CATEGORY####################################
    with open(path+"gem_internal_ver2_SVM.pkl", 'rb') as file:
        model_svm = pickle.load(file)

    print("Model has been loaded succesfully")

    print("Model has "+str(len(model_svm.classes_))+" no of class")
    return model_svm
model_svm=load_svm() 

###############################################################################

def lemmatize_text(text):
    #l= [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
    #return ' '.join(l)
    #return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
    return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])

def level1_preprocess(orig_str):
    
    ########REMOVE DIGITS##########3
    mystr_digits_removed=''.join('' if c.isdigit() else c for c in orig_str)
    #print("digits removed=====",mystr_digits_removed)

    #############REMOVE SPECIAL CHARS###############


    #mystr_special_removed= re.sub('[^a-zA-Z0-9 \n\.]', '', my_str)
    mystr_special_removed= re.sub('[^a-zA-Z0-9 \n\.]', '',mystr_digits_removed)
    #print("special chars removed=======",mystr_special_removed)    



    mystr_special_removed=mystr_special_removed.replace('.',' ')
    mystr_special_removed = mystr_special_removed.replace('  ', ' ') 
    mystr_special_removed=mystr_special_removed.strip()
    
    #print("dot removed=====",mystr_special_removed)  

    ##############REMOVE STOPWORDS##############
       

  
    

    
    word_tokens = word_tokenize(mystr_special_removed)

    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    #print(filtered_sentence)
    filter_sentence=(' '.join(filtered_sentence))


    ###################DO LEMMATIZATION##############################
    t=lemmatize_text(filter_sentence)
    
    return t
    
    
def level2_preprocess(orig_str):
    
    ########REMOVE DIGITS##########3
    #mystr_digits_removed=''.join('' if c.isdigit() else c for c in orig_str)
    #print("digits removed=====",mystr_digits_removed)

    #############REMOVE SPECIAL CHARS###############


    #mystr_special_removed= re.sub('[^a-zA-Z0-9 \n\.]', '', my_str)
    mystr_special_removed= re.sub('[^a-zA-Z0-9 \n\.]', '',orig_str)
    #print("special chars removed=======",mystr_special_removed)    



    mystr_special_removed=mystr_special_removed.replace('.',' ')
    mystr_special_removed = mystr_special_removed.replace('  ', ' ') 
    mystr_special_removed=mystr_special_removed.strip()
    
    #print("dot removed=====",mystr_special_removed)  

    ##############REMOVE STOPWORDS##############
       

  
    

    
    word_tokens = word_tokenize(mystr_special_removed)

    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    #print(filtered_sentence)
    filter_sentence=(' '.join(filtered_sentence))


    ###################DO LEMMATIZATION##############################
    t=lemmatize_text(filter_sentence)
    
    return t

    
    
def call_json():
    return {'message':'hello world'}
########################################################################################################## 
def call_svm(string):
    try:
        if string:
            predicted=''
            prob=0.0
            ###############level 1 predict##############
            print("ORIGINAL STRING==========",string)
            l1_string=level1_preprocess(string)
            print("LEVEL1 STRING==========",l1_string)
            p1,p2=model_amazon(model_bert,string)
            print("bert=========",p2)
            test_features=[l1_string]
            pred_svm_level1=model_level1.predict(test_features)
            prb_list_svm_level1=model_level1.predict_proba(test_features)
            pb_svm_level1=[]
            for i in range(0,len(prb_list_svm_level1)):
                pb_svm_level1.append(round(prb_list_svm_level1[i].max(),2))
            probability_list=list(model_level1.predict_proba([test_features[0]])[0])
            category_list=list(model_level1.classes_)
            dataFrame=pd.DataFrame()
            dataFrame['Category']=category_list
            dataFrame['Probability']=probability_list
            dataFrame=dataFrame.sort_values('Probability',ascending=False)
            dataFrame['Probability']=list(round(dataFrame['Probability'],2))
            
            top3_cat=dataFrame['Category'][:3].values
            top3_prb=dataFrame['Probability'][:3].values
            print("TOP 3 LEVEL1====================",top3_cat)
            print("TOP 3 PROB LEVEL1 ====================",top3_prb)
            print("PREDICTED LEVEL 1 CATEGORY===========================",pred_svm_level1[0])
            if p2[0]<pb_svm_level1[0].astype(float):
                print("========SVM WINS===========")
                if (pb_svm_level1[0].astype(float))>=0.5:
                  if pred_svm_level1[0]=="individual_category":

                      l2_string=level2_preprocess(string)
                      print("LEVEL2 STRING==========",l2_string)
                      test_features_l2=[l2_string]
                      pred_svm_level2_indi=model_level2_individual_category.predict(test_features_l2)
                      prb_list_svm_level2_indi=model_level2_individual_category.predict_proba(test_features_l2)
                      pb_svm_level2_indi=[]
                      for i in range(0,len(prb_list_svm_level2_indi)):
                          pb_svm_level2_indi.append(round(prb_list_svm_level2_indi[i].max(),2))
                      predicted=pred_svm_level2_indi[0]
                      prob=pb_svm_level2_indi[0]
                      if prob>=0.5:
                        predicted=pred_svm_level2_indi[0]
                        prob=pb_svm_level2_indi[0]
                      else:
                        predicted="others"
                        #predicted=pred_svm_level1[0]
                        prob=pb_svm_level2_indi[0]

                  elif pred_svm_level1[0] in club_cat:
                      name=data_file[pred_svm_level1[0]]
                      model=map_subclass[name]
                      l2_string=level2_preprocess(string)
                      print("LEVEL2 STRING==========",l2_string)
                      test_features_l2=[l2_string]
                      pred_svm_level2_club=model.predict(test_features_l2)
                      prb_list_svm_level2_club=model.predict_proba(test_features_l2)
                      pb_svm_level2_club=[]
                      for i in range(0,len(prb_list_svm_level2_club)):
                          pb_svm_level2_club.append(round(prb_list_svm_level2_club[i].max(),2))
                      predicted=pred_svm_level2_club[0]
                      prob=pb_svm_level2_club[0]
                      if (prob.astype(float))>=0.5:
                        predicted=pred_svm_level2_club[0]
                        prob=pb_svm_level2_club[0]
                      else:
                        predicted="others"
                        prob=pb_svm_level2_club[0]
                  else:
                      predicted=pred_svm_level1[0]
                      prob=pb_svm_level1[0]


                else:
                    predicted="others"
                    #predicted=pred_svm_level1[0]
                    prob=pb_svm_level1[0]
            if p2[0]>pb_svm_level1[0].astype(float):
                print("===============BERT WINS==============")
                if (p2[0].astype(float))>=0.5:
                  if p1[0][0]=="individual_category":

                      l2_string=level2_preprocess(string)
                      print("LEVEL2 STRING==========",l2_string)
                      test_features_l2=[l2_string]
                      pred_svm_level2_indi=model_level2_individual_category.predict(test_features_l2)
                      prb_list_svm_level2_indi=model_level2_individual_category.predict_proba(test_features_l2)
                      pb_svm_level2_indi=[]
                      for i in range(0,len(prb_list_svm_level2_indi)):
                          pb_svm_level2_indi.append(round(prb_list_svm_level2_indi[i].max(),2))
                      predicted=pred_svm_level2_indi[0]
                      prob=pb_svm_level2_indi[0]
                      if prob>=0.5:
                        predicted=pred_svm_level2_indi[0]
                        prob=pb_svm_level2_indi[0]
                      else:
                        predicted="others"
                        #predicted=pred_svm_level1[0]
                        prob=pb_svm_level2_indi[0]

                  elif p1[0][0] in club_cat:
                      name=data_file[p1[0][0]]
                      model=map_subclass[name]
                      l2_string=level2_preprocess(string)
                      print("LEVEL2 STRING==========",l2_string)
                      test_features_l2=[l2_string]
                      pred_svm_level2_club=model.predict(test_features_l2)
                      prb_list_svm_level2_club=model.predict_proba(test_features_l2)
                      pb_svm_level2_club=[]
                      for i in range(0,len(prb_list_svm_level2_club)):
                          pb_svm_level2_club.append(round(prb_list_svm_level2_club[i].max(),2))
                      predicted=pred_svm_level2_club[0]
                      prob=pb_svm_level2_club[0]
                      if (prob.astype(float))>=0.5:
                        predicted=pred_svm_level2_club[0]
                        prob=pb_svm_level2_club[0]
                      else:
                        predicted="others"
                        prob=pb_svm_level2_club[0]
                  else:
                      predicted=pred_svm_level1[0]
                      prob=pb_svm_level1[0]


                else:
                    predicted="others"
                    #predicted=pred_svm_level1[0]
                    prob=pb_svm_level1[0]
            


            
    
            
            dict_two={predicted:prob.astype(float)
                #top3_cat[0]:top3_prb[0].astype(float)#,
                #top3_cat[1]:top3_prb[1],
                #top3_cat[2]:top3_prb[2]
            }
            return dict_two
        else:
            return {}
    except:
        return {}


'''@app.route('/')
def upload_f():
    return render_template('index.html')'''
'''@app.route('/ProductList', methods=['POST','GET'])
def processString():
    jsonInput = request.data 
    dictInput = json.loads(jsonInput)
    txt = dictInput['searchedString']
    modelName = dictInput['selectedModel']
    print("InputString-processString",txt)
    global InputString
    InputString = txt
    specialChars='!#$%^&*()/'
    for i in specialChars:
        txt=txt.replace(i,'') ##########no space
    txt=txt.replace(',',' ')
    txt=txt.lower()
    #print(type(modelName))
    if modelName == 'None':
        topCategories, productList= none_module(txt)
        productList = dict({"blankArray" : True if (len(productList) == 0) else False, "isSpellCorrect":True,"searchedString" : InputString,"productList" : productList,"topCategories":topCategories})
    elif modelName == 'SVM':
        topCategories, productList,catt,s = spell_check_module(txt,modelName)
        productList = dict({"blankArray" : True if (len(productList) == 0) else False, "isSpellCorrect":isSpellCorrect,"searchedString" : s,"productList" : productList,"topCategories":topCategories})
    elif modelName == 'BERT':
        topCategories, productList,catt,s = spell_check_module(txt,modelName)
        productList = dict({"blankArray" : True if (len(productList) == 0) else False, "isSpellCorrect":isSpellCorrect,"searchedString" : s,"productList" : productList,"topCategories":topCategories})
    else:
        topCategories, productList,catt,s = spell_check_module(txt,modelName)
        productList = dict({"blankArray" : True if (len(productList) == 0) else False, "isSpellCorrect":isSpellCorrect,"searchedString" : s,"productList" : productList,"topCategories":topCategories})
    return productList'''
######################################################################################################################
#Funtion to correct the misspelt Input string
def spell_check_module(test_string):
    find_num=re.findall('\d+', test_string)
    if len(find_num)!=0:
        z=1
        for i in find_num:
            if z==1:
                test_string2=test_string.replace(i,"")
                z=z+1
            else:
                test_string2=test_string2.replace(i,"")
       
        print(test_string2)
        split_text=test_string2.split()
        print("split========",split_text)
        correct=[]
        flag=[]
        st=' '
        dd={}
        for i in split_text:
            dd.setdefault(i,[])
            lm_models = arpa.loadf(path+"spell_check_lm.arpa")
            arpa_model = lm_models[0]
            sym_spell = SymSpell(max_dictionary_edit_distance=2)
            sym_spell.create_dictionary(path+"all_cat_ss_corpus_v2.txt")
            sym_spell.lookup(i,Verbosity.ALL)
            spell_list=[each_sym.term for each_sym in sym_spell.lookup(i,Verbosity.CLOSEST,max_edit_distance=2,include_unknown=True)]
            if spell_list[0]==i:
                flag.append(0)
                dd[i].append(i)
            else:
                flag.append(1)
                dd[i].append(spell_list[0])
       
        print("dict mapping===================",dd)
        for word, initial in dd.items():
            test_string = test_string.replace(word,initial[0])
        print("address@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",test_string)
        s = test_string
        #return test_string,flag
       
    else:
        split_text=test_string.split()
        correct=[]
        flag=[]
        s=' '
        for i in split_text:
            lm_models = arpa.loadf(path+"spell_check_lm.arpa")
            arpa_model = lm_models[0]
            sym_spell = SymSpell(max_dictionary_edit_distance=2)
            sym_spell.create_dictionary(path+"all_cat_ss_corpus_v2.txt")
            sym_spell.lookup(i,Verbosity.ALL)
            spell_list=[each_sym.term for each_sym in sym_spell.lookup(i,Verbosity.CLOSEST,max_edit_distance=2,include_unknown=True)]
            if spell_list[0]==i:
                flag.append(0)
                correct.append(i)
            else:
                flag.append(1)
                correct.append(spell_list[0])
        s=s.join(correct)
        print(s,flag)
        #return s,flag
    
    if 1 in flag:
        global isSpellCorrect
        #print("in isSpellCorrect",isSpellCorrect)
        isSpellCorrect = False
    else:
        #global isSpellCorrect
        isSpellCorrect = True
    print("isSpellCorrect",isSpellCorrect)
    #return s,correct,flag
    #if modelName == 'SVM':
        #topCategories,productList,catt = predict_svm(s)
    #elif modelName == 'BERT':
        #m=load_bert() 
        #topCategories,productList,catt = model_amazon(s)
    #else:
        #topCategories,productList,catt = predict_ensemble(s)
    #print("return-spellcheck",topCategories,productList)
    #return topCategories,productList,catt,s
    return s

#########################################################################################
#Function to predict the category of the Input string
def predict_svm(txt):
    print("===============ORIGINAL TEXT=================",txt)
    txt=level1_preprocess(txt)
    print("=================PREPROCESSED TEXT====================",txt)
    
    ######################SVM MODEL FOR SEARCH ENGINE CATEGORY####################################
    #with open(path+"svm_model_search_ver3.pkl", 'rb') as file:
       # model_svm = pickle.load(file)

    #print("Model has been loaded succesfully")

    print("Model has "+str(len(model_svm.classes_))+" no of class")
    user_input=txt
    test_features=[user_input]
    #print(test_features)
    ##############MODEL CATEGORY PREDICTION############################
    pred_svm=model_svm.predict(test_features)
    prb_list_svm=model_svm.predict_proba(test_features)
    pb_svm=[]
    for i in range(0,len(prb_list_svm)):
        pb_svm.append(round(prb_list_svm[i].max(),2))
    print("---{test1}---------{class1}---------{prob}".format(test1=test_features,class1=pred_svm,prob=pb_svm))
    #category=pred_svm[0]
    category=''
    print("Predicted Category",category)
    if (pb_svm[0].astype(float))<0.50:
        category=''
        topCategories = []
        productList = []
    else:
        #print("1")
        #return category
        category=pred_svm[0]
        topCategories = top_3_cat(category);
        productList = predict_ner(category,txt)
    #print("return-predictSVM",topCategories,productList)
    #print("------------topCategories-------------",topCategories)
    #print("------------category-------------",category)
    #print("------------productList-------------",productList)
    return topCategories,productList,category,pb_svm[0].astype(float)
def load_ner():
    ####################MODEL FOR NER FEATURE EXTRACTION FROM STRING#######################
    with open(path+"svm_model2.pkl", 'rb') as file:
        model_ner = pickle.load(file)

    print(" NER Model has been loaded succesfully")

    print("Model has "+str(len(model_ner.classes_))+" no of class")
    return model_ner
    
model_ner=load_ner()    
#Function to extract the feature from Input String
def predict_ner(category,txt):
    user_input=txt
    test_features=[user_input]
    #print(test_features)
    ####################MODEL FOR NER FEATURE EXTRACTION FROM STRING#######################
    #with open(path+"svm_model2.pkl", 'rb') as file:
      #  model_ner = pickle.load(file)

    #print("Model has been loaded succesfully")

    print("Model has "+str(len(model_ner.classes_))+" no of class")
    all_candidates=[]
    aa=[]
    for i in range(1,4):
        aa.append(generate_ngrams(test_features[0], n=i))
    all_candidates=[item for sublist in aa for item in sublist]
    tt=[]
    tag=[]
    #print("all_candidates",all_candidates)
    for j in all_candidates:
        test_features=[j]
        tt.append(j)
    

        pb_ner=[]
        pred_ner=model_ner.predict(test_features)
        prb_list_ner=model_ner.predict_proba(test_features)
        tag.append(pred_ner[0])
        for i in range(0,len(prb_list_ner)):
            pb_ner.append(round(prb_list_ner[i].max(),2))
    
        print("---{test1}---------{class1}---------{prob}".format(test1=j,class1=pred_ner,prob=pb_ner))
    dd={}
    for i in set(tag):
        dd.setdefault(i,[])
        for j in range(len(tt)):
            if tag[j]==i:
                dd[i].append(tt[j])
    ner2=pd.read_csv(path+"named_entity_flipkart2.csv")
    features=set(list(ner2['TAGS']))
    a=[]
    b=[]
    for i in features:
        tags_ner=list(ner2[ner2['TAGS']==i].iloc[:,0])
    
        for i in dd.keys():
            list1=dd[i]
            for j in list1:
                if j in tags_ner:
                    #print("yes")
                    b.append(j)
                    a.append(i)
                    #print("2")
    #return a,b
    productList = read_catalogue_data(category,a,b)
    #print("return-predict_ner",productList)
    return productList
    


def generate_ngrams(s, n):
    # Convert to lowercases
    
    # Break sentence in the token, remove empty tokens
    tokens = [token for token in s.split(" ") if token != ""]
    
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    #print("3")
    return [" ".join(ngram) for ngram in ngrams]

#Function to return the product list for given Input Category
def read_catalogue_data(category,a,b):
    catalogue=pd.read_excel(path+"catalogue_"+category+".xlsx",engine='openpyxl')
    col=list(catalogue.columns)
    len_of_catalogue=catalogue.shape[0]
    feature_dict={}
    for i in set(a):
        feature_dict.setdefault(i,[])
        for j in range(len(b)):
            if a[j]==i:
                feature_dict[i].append(b[j])
    z=[]
    cc=1
    boss=0
    for i in feature_dict.keys():
        cap=i
        i=i.lower()
        #print("i",i)
        if i in col:
            list1=feature_dict[cap]
            #print("yes")
            boss=1
            #print(list1)
            cc=1
            for k in list1:
                filter1=catalogue[i]==k
                if cc==1:
            
                    filter_initial=filter1
            
                #print((filter_initial))
                    cc=cc+1
                else:
                    filter1=(filter_initial | filter1)
                    filter_initial=filter1
                #print(filter1)
            z.append(list(filter_initial))
        else:
            print("not a feature")
    if boss!=0:
        q=[]
        for i in z:
            q.append(np.array(i))
        orr=[True]*len_of_catalogue
        orr=np.array(orr)
        for i in q:
            orr=orr & i
    else:
        orr=[True]*len_of_catalogue
        orr=np.array(orr)
    orr=list(orr) ###########IMPORTANT STEP
    catalogue['flag']=orr
    path1=catalogue[catalogue['flag']==True].loc[:,['title','brand','image']]
    #print(path1)
    productList = path1.to_json(orient="records")
    productList = json.loads(productList)
    #print(type(productList))
    #print("return-read_catalogue_data",productList)
    return productList ##############path1 is list of all valid images ######################
##########################################################################################################
def top_3_cat(category):
    #####"category" here is the category which function "predict_svm" returns [category=predict_svm("smart televisions")]#####
    dict1={}
    top3=[]
    categoryList = []
    dict1["laptop-notebook"]=['all in one pc','computer workstation','desktop computers']
    dict1["laptop bag"]=['canvas bag','bagpacks','bags luggage']
    dict1['smart tv']=['Television TV','Smart Class Equipment With Digital Contents Software','Smart Rack']
    dict1['smart phone']=['Smart Phone With Mdm/Emm/Sdk','smart watch with sim card','smart band']
    list1=list(dict1.keys())
    if category in list1:
        for j in dict1[category]:
            top3.append(j)
    else:
        print("sorry wrong category.....no products to display")
    for i in range(0,len(top3)):
        categoryList.append({"id":i+1,"category" :top3[i]})
    return categoryList 
    
###################################################################################################################
#########-----predict-bert--------################
def model_amazon(txt):
    print("===============ORIGINAL TEXT=================",txt)
    txt=level1_preprocess(txt)
    print("=================PREPROCESSED TEXT====================",txt)
    predd=[]
    list1=[txt]
    for i in list1:
        X_train=[i]
        inputs=create_input_array(X_train)
        #encoder = LabelEncoder()
        #encoder.fit(map_cat)
        #encoded_Y = encoder.transform(map_cat)
        '''encoder = LabelEncoder()
        encoder.fit(y_train)
        encoded_Y = encoder.transform(y_train)'''
        print('\n')
        pred = model_bert.predict(inputs)
        b=list(encoder.inverse_transform(np.apply_along_axis(np.argmax,1,pred)))
        #print(i,b)
        pp=np.apply_along_axis(np.max,1,pred)
        #print("String tested==== \'{title}\' and CLASS PREDICTED ===={catt}".format(title = i,catt = b))
        print("String tested==== \'{title}\' and CLASS PREDICTED ===={catt}---{prob}".format(title = i,catt = b,prob=pp))
        predd.append(b)
        category=''
        #return predd[0][0]
        if (pp[0].astype(float))<0.50:
            category=''
            topCategories = []
            productList = []
        else:
            #print("1")
            #return category
            category=predd[0][0]
            topCategories = top_3_cat(predd[0][0]);
            productList = predict_ner(predd[0][0],txt)

        #topCategories = top_3_cat(predd[0][0]);
        #productList = predict_ner(predd[0][0],txt)
        #return topCategories,productList,predd[0][0]
        return topCategories,productList,category,pp[0].astype(float)
#########################################################################################################################


#@app.route('/queryget', methods = ['GET'])
def call_modules_get(test):
        try:
            print('here')
            #string = request.args.get('string')
            #inputjson = request.data
            #input_json = json.loads(inputjson)
            #string = input_json['query']
            string=test
            print("Input string", string)
            if string:
                string = string.lower()
                
                
                
                #dict_svm = predict_svm(string)
                #print("corrected_query=========",corrected_query)
                #dict_svm = call_svm(cleaned_string)
                topCategories_svm,productList_svm,category_svm,prob_svm=predict_svm(string)
                print("------------SVM-------------",topCategories_svm,category_svm,productList_svm[0:3],prob_svm)
                
                topCategories_bert,productList_bert,category_bert,prob_bert=model_amazon(string)
                print("------------BERT-------------",topCategories_bert,category_bert,productList_bert[0:3],prob_bert)
                if prob_svm>=prob_bert:
                    print("--------------SVM WINS-------------",prob_svm,prob_bert)
                    topCategories,productList,category,prob=topCategories_svm,productList_svm,category_svm,prob_svm
                else:
                    print("--------------BERT WINS-------------",prob_svm,prob_bert)
                    topCategories,productList,category,prob=topCategories_bert,productList_bert,category_bert,prob_bert
                
                final_pred_lis = []
                #category = list(dict_svm.keys())[0]
                #score=dict_svm[category]
                
                if category!="others":
                  #id_=cat_id[cat_id['Category Name']==category]['Category ID'].values[0]
                  final_pred_lis.append({'name':category,'productList':productList,"topCategories":topCategories})
                else:
                  #id_="others"
                  final_pred_lis.append({'name':'others','productList':'No products to display',"topCategories":"None"})
                  
                #final_pred_lis.append({'name':category,'score':score})
                
                #print(json.dumps(str({'spell_check':{'isWronglySpelt':flag,'modifiedQuery':corrected_query},'categories':final_pred_lis,'tags':brand})))
                return json.dumps({"categories":final_pred_lis})
            else:
                print("----------- I m in none case of string-------------------")
                final_pred_lis = []
                final_pred_lis.append({'name':'','productList':'',"topCategories":''})
                return json.dumps({"categories":final_pred_lis})
                
                #return jsonify({'spell_check':{'isWronglySpelt':flag,'modifiedQuery':corrected_query},'categories':final_pred_lis,'tags':brand})
        except Exception as e:
            print(e)
            jsonContent = {'Error Code': "400","Error Text":"Bad Request"}
            jsonContent = json.dumps(jsonContent)
            return(jsonContent)
            #return jsonify({'Error Code': "400","Error Text":"Bad Request"})


#if __name__ == '__main__':
 #   app.run(host='localhost',port='8080',debug=True,use_reloader=False)

#making function
def predict():
    SEARCH_STRING= userInput.get()
    
    try:
        SEARCH_STRING = str(SEARCH_STRING)
        
        
        dc = {'SEARCH_STRING':[SEARCH_STRING]}
        print("============dc============",dc)
        test=pd.DataFrame.from_dict(dc)
        ####################
        InputString = dc['SEARCH_STRING'][0]
        txt=InputString
        specialChars='!#$%^&*()/'
        for i in specialChars:
            txt=txt.replace(i,'') ##########no space
        txt=txt.replace(',',' ')
        txt=txt.lower()
        spell_str=spell_check_module(txt)
        print("================= spell check=================",spell_str)
        ###########################################
        js=call_modules_get(spell_str)
        #print("js====",json.loads(js))
        y=json.loads(js)
        #print("js====",json.loads(js))
        #for label in all_labels:
           # label.destroy()
        '''c=0
        pic2=['Capture_se2.PNG']
        pic=pic2*10
        for number in range(len(pic)):
            #filename = f"images/{pic_name}/{pic_name}{number+1}.jpeg" # f-string (Python 3.6+)
            filename = path+pic[number]# (Python 2.7, 3.0+)

            image = ImageTk.PhotoImage(Image.open(filename))

            #label = Label(image=image)
            #label.photo = image   # assign to class variable to resolve problem with bug in `PhotoImage`

            if (number%5==0):


                c=c+1
                print("yes",number,c)
                try:
                    #imageEx = PhotoImage(file = path+pic[number])
                    #label = Label(colorLog,image=image,text="anshu")
                    label = Label(image=image,text="anshu")
                    label.photo = image
                    label.grid(row=c, column=(number%5))
                    label['compound']=tk.Top
                    #Label(colorLog, image=imageEx).grid(row=c, column=(number%5))
                except:
                    print("Image not found")
                #label.grid(row=c, column=(number%5))

            else:
                print("no",number,c)
                #label.grid(row=c, column=(number%5))
                try:
                    #imageEx = PhotoImage(file = path+pic[number])
                    #label = Label(colorLog,image=image,text="anshu")
                    label = Label(image=image,text="anshu")
                    label.photo = image
                    label.grid(row=c, column=(number%5))
                    label['compound']=tk.Top
                    #Label(colorLog, image=imageEx).grid(row=c, column=(number%5))
                except:
                    print("Image not found")

            #https://stackoverflow.com/questions/62910727/displaying-multiple-images-side-by-side-with-tkinter            
        
            
            
            
            
        
        #all_labels.append(label)'''

        
        #print("output============",js)
        pr=[]
        for i in y["categories"][0]["productList"]:
            pr.append(i['title'])
        #scrollbar = Scrollbar(colorLog)
        #scrollbar.pack( side = RIGHT, fill = Y )   
        colorLog.delete(0.0,END)
        colorLog.insert(END,"Did you mean"+" "+str(spell_str)+"\n")
        colorLog.insert(END, str(y["categories"][0]["name"])+"\n")
        #for q in pr:
            #colorLog.insert(END,str(q)+"\n")
        #colorLog.pack( side = LEFT, fill = BOTH )
        #scrollbar.config( command = mylist.yview )
        #colorLog.insert(0.0,str(y["categories"][0]["productList"])+"\n")
        colorLog.insert(0.0,str(pr)+"\n")
        
        #colorLog.insert(0.0,str(y["categories"][0]["topCategories"])+"\n")
        #lbl11['text'] = [y["categories"][0]['name'],y["categories"][0]['score'],y["categories"][0]['id']]
        
    except ValueError:
        messagebox.showinfo("Alert Message", "Enter Properly, bro!") # this code to make alert messages when we wrong input data


'''wdw = tk.Tk()
wdw.title("NLP Engine predict!") #mengubah nama jendela
#MonthlyCharges
inp1 = tk.Entry(wdw)
inp1.insert(0,'Type your text here................')
inp1.grid(row=1,column=1)
lbl1 = tk.Label(wdw, text="ENTER THE SEARCH STRING")
lbl1.grid(row=1,column=0)

btn = tk.Button(wdw, text = "Click to predict!", command=predict)
btn.grid(row=12,columnspan=2)
lbl11 = tk.Label(wdw, text="Result ...")
lbl11.grid(row=11,columnspan=2)
wdw.mainloop()'''

root = Tk() #Makes the window
root.wm_title("Search Engine Optimization") #Makes the title that will appear in the top left
root.config(background = "#FFFFFF")





#Left Frame and its contents
leftFrame = Frame(root, width=200, height = 600)
leftFrame.grid(row=0, column=0, padx=10, pady=2)

Label(leftFrame, text="NLP Search Engine Optimization!:").grid(row=0, column=0, padx=10, pady=2)

#Instruct = Label(leftFrame, text="1\n2\n2\n3\n4\n5\n6\n7\n8\n9\n")
#Instruct.grid(row=1, column=0, padx=10, pady=2)

try:
    imageEx = PhotoImage(file = path+"Capture_se2.PNG")
    Label(leftFrame, image=imageEx).grid(row=2, column=0, padx=10, pady=2)
except:
    print("Image not found")

#Right Frame and its contents
rightFrame = Frame(root, width=200, height = 600)
rightFrame.grid(row=0, column=1, padx=10, pady=2)
lbl1 = Label(rightFrame, text="ENTER THE SEARCH STRING")
lbl1.grid(row=0,column=2,padx=10, pady=2)
circleCanvas = Canvas(rightFrame, width=300, height=100, bg='light blue')
circleCanvas.grid(row=0, column=0, padx=10, pady=2)

btnFrame = Frame(rightFrame, width=200, height = 200)
btnFrame.grid(row=1, column=0, padx=10, pady=2)

btn = Button(btnFrame, text = "Click to predict!", command=predict)
btn.grid(row=12,columnspan=2)

colorLog = Text(rightFrame, width = 30, height = 20, takefocus=0)
colorLog.grid(row=2, column=0, padx=10, pady=2)
################# new ########

#######################
#redBtn = Button(btnFrame, text="Red", command=redCircle)
#redBtn.grid(row=0, column=0, padx=10, pady=2)

#yellowBtn = Button(btnFrame, text="Yellow", command=yelCircle)
#yellowBtn.grid(row=0, column=1, padx=10, pady=2)

#greenBtn = Button(btnFrame, text="Green", command=grnCircle)
#greenBtn.grid(row=0, column=2, padx=10, pady=2)

userInput=Entry(rightFrame,width=45)
userInput.insert(0,'')
userInput.grid(row=0, column=0, padx=10, pady=2)

#get the text inside of userInput
userInput.get()

root.mainloop() #start monitoring and updating the GUI


