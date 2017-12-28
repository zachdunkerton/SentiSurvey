import pandas as pd 
import re 
import pickle
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

train = pd.read_csv("./data/train.csv")
train_docs = train.iloc[0:len(train),[1,2]]
senti = train.iloc[0:len(train), [3]]

translator = str.maketrans('', '', string.punctuation)

def method1(word_data, sentiment_data):
    #read in the lists of positive and negative words
    fileReader = open("./data/negative-words.txt", "r")
    lines = fileReader.read()
    fileReader.close()
    lines = lines.split('\n')

    #create a list of all the negative words
    negative_words = []
    for word in lines:
        negative_words.append(str(word))

    fileReader = open("./data/positive-words.txt", "r")
    lines = fileReader.read()
    fileReader.close()
    lines = lines.split('\n')
    #create a list of all the positive words
    positive_words = []
    for word in lines:
        positive_words.append(str(word))

    count = 0
    #for each document, split into individual workds
    while count < len(word_data):
        doc = re.split('\W+', word_data[count])
        #for each individual word, find it's overall document sentiment in the sentiment labels list
        for words in doc:
            if sentiment_data[count] == -1:
                #if the document is negative, search for it in the negative words list
                if words in negative_words:
                    #if it is in the negative words list, find its definitions 
                    for y in method1NegWords(words):
                        #add these additional features to that word's document
                        word_data[count] += " " + y
            #if the document is positive, search for it in the positive words list
            elif sentiment_data[count] == 1:
                if words in positive_words:
                     #if it is in the positive words list, find its definitions 
                    for y in method1PosWords(words):
                        #add these additional features to that word's document
                        word_data[count] += " " + y
        count +=1
    return word_data

def method1NegWords(word):
    append = []
    #look up all of the words definitons in WordNet
    word_def =  wn.synsets(word)
    count = 0
    for words in word_def:
        #look up each definition in SenitWordNet
        senti = swn.senti_synset(words.name())
        #sentiment score thresholds
        if senti.neg_score() > 0.5 and senti.pos_score() ==0:
            #if that definition meets the sentiment threshold, add that word and the list of things to be appended
            append.append(str(words).split("'")[1].split(".")[0] + " ")
            append.append(words.definition() + " ")
            count+=1
    return append

def method1PosWords(word):
    append = []
    #look up all of the words definitons in WordNet
    word_def =  wn.synsets(word)
    count = 0
    for words in word_def:
        #look up each definition in SenitWordNet
        senti = swn.senti_synset(words.name())
        #sentiment score thresholds
        if senti.pos_score() > 0.5 and senti.neg_score() ==0:
            #if that definition meets the sentiment threshold, add that word and the list of things to be appended
            append.append(str(words).split("'")[1].split(".")[0] + " ")
            append.append(words.definition() + " ")
            count+=1
    return append
def method2(word_data, sentiment_data):
    #read in the lists of positive and negative words
    fileReader = open("./data/negative-words.txt", "r")
    lines = fileReader.read()
    fileReader.close()
    lines = lines.split('\n')

    #create a list of all the negative words
    negative_words = []
    for word in lines:
        negative_words.append(str(word))

    fileReader = open("./data/positive-words.txt", "r")
    lines = fileReader.read()
    fileReader.close()
    lines = lines.split('\n')
    #create a list of all the positive words
    positive_words = []
    for word in lines:
        positive_words.append(str(word))

    count = 0
    #for each document, split into individual workds
    while count < len(word_data):
        doc = re.split('\W+', word_data[count])
        #for each individual word, find it's overall document sentiment in the sentiment labels list
        for words in doc:
            if sentiment_data[count] == -1:
                #if the document is negative, search for it in the negative words list
                if words in negative_words:
                    #if it is in the negative words list, find its definitions 
                    for y in method2NegWords(words):
                        #add these additional features to that word's document
                        word_data[count] += " " + y
            #if the document is positive, search for it in the positive words list
            elif sentiment_data[count] == 1:
                if words in positive_words:
                     #if it is in the positive words list, find its definitions 
                    for y in method2PosWords(words):
                        #add these additional features to that word's document
                        word_data[count] += " " + y
        count +=1
    return word_data

def method2NegWords(word):
    #we are looking for the definition with the highest negative score for a given word
    #look up word in WordNet
    word_def =  wn.synsets(word)
    max = 0
    maxWord = ""
    #iterate through all definitions
    for words in word_def:
        #look up word in SentiWord net
        senti = swn.senti_synset(words.name())
        #compare score to current max score
        if senti.neg_score()>max:
            #if it is greater, set the new max word
            maxWord = words.name()
            max = senti.neg_score()
    #if the word has a definition with a negative score, append synonyms and definition
    if(maxWord != ""):
        x = wn.synset(maxWord).definition() 
        append.append(x)
        x = wn.synsets(maxWord)
        split = maxWord.split('.')
        append.append(split[0])
        x = wn.synsets(split[0])
        synonym = ""
        for stuff in x:
            if stuff.name().split(".")[0] != synonym:
                append.append(stuff.name().split(".")[0])
                synonym = stuff.name().split(".")[0]
    return append

def method2PosWords(word):
    #we are looking for the definition with the highest positive score for a given word
    #look up word in WordNet
    word_def =  wn.synsets(word)
    max = 0
    maxWord = ""
    #iterate through all definitions
    for words in word_def:
        #look up word in SentiWord net
        senti = swn.senti_synset(words.name())
        #compare score to current max score
        if senti.pos_score()>max:
            #if it is greater, set the new max word
            maxWord = words.name()
            max = senti.pos_score()
    #if the word has a definition with a positive score, append synonyms and definition
    if(maxWord != ""):
        x = wn.synset(maxWord).definition() 
        append.append(x)
        x = wn.synsets(maxWord)
        split = maxWord.split('.')
        append.append(split[0])
        x = wn.synsets(split[0])
        synonym = ""
        for stuff in x:
            if stuff.name().split(".")[0] != synonym:
                append.append(stuff.name().split(".")[0])
                synonym = stuff.name().split(".")[0]
    return append

def method3(word_data, sentiment_data):
    count = 0
    #iterate through every document 
    while count < len(word_data):
        #negative documents
        if sentiment_data[count] == -1:
            #splits the documents into individual words
            doc = re.split('\W+', word_data[count])
            #holds score of highest word
            max = 0
            #holds highest word
            maxWord = ""
            #iterate through each word in the document
            for words in doc:
                #look up word in WordNet
                word_def =  wn.synsets(words)
                #iterate through each definition
                for defs in word_def:
                    #look up definition in SentiWordNet
                    senti = swn.senti_synset(defs.name())
                    #if it has the highest negative score, make it the max
                    if senti.neg_score()>max:
                        maxWord = defs.name()
                        max = senti.neg_score()
            #append definition
            if max >0:
                #append actual definition
                word_data[count]+= " " + wn.synset(maxWord).definition() 
                x = wn.synsets(maxWord.split('.')[0])
                #append synonyms but try to avoid duplicates
                synonym = ""
                #iterate thro
                for stuff in x:
                    if stuff.name().split(".")[0] != synonym:
                        #append synonyms
                        word_data[count]+= " " + stuff.name().split(".")[0]
                        synonym = stuff.name().split(".")[0]
            count +=1
        #same process but for positive documents
        elif sentiment_data[count] == 1:
            doc = re.split('\W+', word_data[count])
            max = 0
            maxWord = ""
            for words in doc:
                word_def =  wn.synsets(words)
                for defs in word_def:
                    senti = swn.senti_synset(defs.name())
                    if senti.pos_score()>max:
                        maxWord = defs.name()
                        max = senti.pos_score()
            #append definition
            if max >0:
                word_data[count]+= " " + wn.synset(maxWord).definition() 
                x = wn.synsets(maxWord.split('.')[0])
                #append synonyms but try to avoid duplicates
                synonym = ""
                for stuff in x:
                    if stuff.name().split(".")[0] != synonym:
                        word_data[count]+= " " + stuff.name().split(".")[0]
                        synonym = stuff.name().split(".")[0]
            count +=1
        else:
            count +=1
    return word_data

#removes any punctionation, and stems the words
def stem(word_data):
    stemmed_data = []
    translator = str.maketrans('', '', string.punctuation)
    for text in word_data:
        text = str(text)
        text = text.translate(translator)
        words = ""
        text_array = re.split('\W+', text)
        stemmer = SnowballStemmer("english")
        for x in text_array:
            x = x.lower()
            words += str(stemmer.stem(x))
            words += " "
        stemmed_data.append(words)
    return stemmed_data

#start with empty lists for the documents and sentiment labels
train_words = []
sentiment = []
i=0
the_id = -1

#read in the documents from the csv
#the data is parsed into phrases, but for this analysis, only want the full document
#each row has a document id, and the first row with that id is the whole document
#for this loop, if the id of that row is equal to the previous one, we already have entered that document
#if its a new id, then its a new document, so we add it and reset the id value

print("reading")
while i < len(train_docs):
    #check to see if its a new id
    if the_id != train_docs.iloc[i,[0]][0]:
        #if it is reset the_id
        the_id = train_docs.iloc[i,[0]][0]
        #this is the new document variable
        doc = train_docs.iloc[i][1]
        #if its a negative or very negative sentiment, add a -1 to the sentiment labels
        if senti.iloc[i][0] == 0 or senti.iloc[i][0] == 1:
            train_words.append(doc)
            sentiment.append(-1)
        #if its a positive or very positive seniment, add positive to the sentiment labels
        elif senti.iloc[i][0] == 3 or senti.iloc[i][0] == 4:
            sentiment.append(1)
            train_words.append(doc)
    i+=1
#get a count of the negative and positive documents so we know how many to add so there are even numbers
negone = 0
one = 0
for x in sentiment:
    if x == -1:
        negone +=1
    elif x == 1:
        one +=1

#read in extra negative documents from separate data source
fileReader = open("./data/extra-negatives.neg", "r")
lines = fileReader.read()
fileReader.close()
#split the data into documents and remove punctuation
lines = lines.split('\n')
translator = str.maketrans('', '', string.punctuation)
index = 0
#for how ever many extra negative documents we need, randomly select documents and add them
print("Adding Extra Negatives")
from random import *
print(one," ",negone)
while index <= (one-negone-1):
    text = lines[randint(0,len(lines)-1)]
    text = text.translate(translator)
    train_words.append(text)
    sentiment.append(-1)
    index+=1

#split this complete set of documents and sentiments into a 70-30 train test split
features_train, features_test, labels_train, labels_test = train_test_split(train_words, sentiment, test_size=0.3, random_state=42)

#one of the three feature addition methods can be called here
#only do this on the train documents
print("Additions")
#features_train = method1(features_train, labels_train)
#features_train = method2(features_train, labels_train)
features_train = method3(features_train, labels_train)




#stem the words
print("Stemming")
features_train = stem(features_train)
features_test = stem(features_test)

#output the preprocessed data
pickle.dump( features_train, open("./trainwords.pkl", "wb") )
pickle.dump( labels_train, open("./trainsentiment.pkl", "wb"))
pickle.dump( features_test, open("./testwords.pkl", "wb") )
pickle.dump( labels_test, open("./testsentiment.pkl", "wb"))
print("Done!")


