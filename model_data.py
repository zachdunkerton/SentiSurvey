#Zachariah Dunkerton
#Towson university
#Sentiment Analysis of Survey Results

#This code reads the already preprocessed data, and trains and tests models based on that data

import pickle
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

#the corpus and the labels are already split by the preprocessing step
#"sentiment" files are the labels
#"words" files are the documents

#reads in the train sentiment data 
sentiment_file_handler = open('./trainsentiment.pkl', "rb")
labels_train = pickle.load(sentiment_file_handler)
sentiment_file_handler.close()

#reads in the test sentiment data
sentiment_file_handler = open('./testsentiment.pkl', "rb")
labels_test = pickle.load(sentiment_file_handler)
sentiment_file_handler.close()

#reads in the training documents
words_file_handler = open('./trainwords.pkl', "rb")
features_train = pickle.load(words_file_handler)
words_file_handler.close()

#reads in the test documents
words_file_handler = open('./testwords.pkl', "rb")
features_test = pickle.load(words_file_handler)
words_file_handler.close()

#combine the train and test words to create full corpus for TFIDF 
both = list(features_test) + list(features_train)

#the TFIDF vectorier converts the documents to numerical values based on term and document frequency
#creates bigrams and trigrams 
vect = TfidfVectorizer(stop_words = "english", ngram_range = {1,3})
#vit the vectorizer based on the full corpus
vect.fit(both)
#transform the individual document sets
features_train = vect.transform(features_train)
features_test = vect.transform(features_test)

#Naive Bayes modle
from sklearn.naive_bayes import MultinomialNB
#create the model
nb = MultinomialNB(alpha=.15,fit_prior=False)
#fit the model
nb.fit(features_train, labels_train)
#predict the test features based on the trained model
pred = nb.predict(features_test)
#print out accuracy metrics
print("Multinomial NB: " , round(accuracy_score(labels_test, pred), 3))
print(classification_report(labels_test, pred, labels=[1,-1]))
#this function dumps the model into a file that can be transfered to web server
#the model can then be reloaded to classify new documents
joblib.dump(vect, "C:/Users/Zach Dunkerton/Desktop/SentiSurvey/server/tfidf.pkl")

#This module finds the best parameters for a model by running it over every possible set of parameters
#This is where the value for alpha came from
#from sklearn.model_selection import GridSearchCV
#from sklearn.naive_bayes import MultinomialNB
#mnnb= MultinomialNB()
#parameters = {'alpha':[0, .1,.2,.3,.4,.5,.6,.7,.8,.9,1,1.5,2,2.5,3,5,10]}
#clf = GridSearchCV(mnnb, parameters)
#clf.fit(features_train, labels_train)
#print("The best parameters are %s with a score of %0.2f"% (clf.best_params_, clf.best_score_))

#same process as above
#Linear SVC is essentially an SVM with a linear kernel
from sklearn.svm import LinearSVC
clf = LinearSVC(C=10)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print("Linear SVM: " , round(accuracy_score(labels_test, pred), 3))
print(classification_report(labels_test, pred, labels=[1,-1]))
