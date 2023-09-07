#!/usr/bin/env python
# coding: utf-8

# In[ ]:


nltk.download()


# In[20]:


import numpy as np # for array # arr = np.array([1, 2, 3, 4, 5])
import pandas as pd # for open file, built dataframe (like excel)


#load dataset
true_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')

#print(fake_news)


# In[21]:


#fake_news.head(10)
#fake_news.tail()
#fake_news["subject"].value_counts() #check different subject and their counts
#print(fake_news["text"])

#categorize the news
fake_news["category"] = 1
true_news["category"] = 0

#combine data and reset index
df = pd.concat([fake_news, true_news]).reset_index(drop=True)


#remove rows with missing text
df.dropna(subset=['text'], inplace=True)

df.tail()


# ## Preprocessing

# In[ ]:


# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer, WordNetLemmatizer

# # Initialize the stemmer and lemmatizer
# porter_stemmer = PorterStemmer()
# wordnet_lemmatizer = WordNetLemmatizer()

# # Define the preprocessing function
# def preprocess_text(text):
#     # Tokenize the text into sentences
#     sentences = sent_tokenize(text)
#     cleaned_sentences = []
    
#     for sentence in sentences:
#         # Tokenize each sentence into words
#         words = word_tokenize(sentence)
        
#         # Remove special characters, convert to lowercase, and filter out stopwords
#         cleaned_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stopwords.words('english')]
        
#         # Apply stemming and lemmatization
#         stemmed_and_lemmatized_words = [wordnet_lemmatizer.lemmatize(porter_stemmer.stem(word)) for word in cleaned_words]
        
#         cleaned_sentences.append(stemmed_and_lemmatized_words)
    
#     # Convert the tokenized sentences back to text
#     processed_text = [' '.join(tokens) for tokens in cleaned_sentences]
    
#     # Combine sentences within the same document
#     combined_text = ' '.join(processed_text)
#     return combined_text


# In[3]:


from nltk.tokenize import word_tokenize
import re

# Tokenize sentence
# Tokenize words
# remove special characters
# convert to lowercase
# filter out stop words
# stemming and lemmatization
# join the words into sentence
# join sentences

# tokenization + remove special character + lowercase
# def clean_and_lower(word):
#     cleaned_word = re.sub(r'[^a-zA-Z]', '', word).lower()
#     return cleaned_word

# # Function to preprocess a single article
# def preprocess_article(article):
#     tokenized = word_tokenize(article)
#     cleaned_sentences = [clean_and_lower(word) for word in tokenized]
#     return cleaned_sentences

#     # Tokenize each sentence into words
#     tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    # Remove special characters from each word and convert to lowercase
#     cleaned_sentences = [[clean_and_lower(word) for word in sentence] for sentence in tokenized_sentences]
   # # filter out stop words

# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))
# df['processed'] = df['processed'].apply(lambda x: [word for word in x if word not in stop_words])

# # stemming and lemmatization

# from nltk.stem import PorterStemmer, WordNetLemmatizer
# porter_stemmer = PorterStemmer()
# wordnet_lemmatizer = WordNetLemmatizer()

# # stemming
# df['processed'] = df['processed'].apply(lambda x: [porter_stemmer.stem(word) for word in x])
# # lemmatizer
# df['processed'] = df['processed'].apply(lambda x: [wordnet_lemmatizer.lemmatize(word) for word in x])

# # Convert the tokenized sentences back to text
# df['processed'] = df['processed'].apply(lambda x: [' '.join(tokens) for tokens in x])
# # Combine sentences within the same document
# df['combined_text'] = df['processed'].apply(lambda x: ' '.join(x))


# In[23]:


from nltk.corpus import stopwords
def preprocess(article):
    stop_words = set(stopwords.words('english'))
    processed_data = []  # Initialize an empty list
    for document in article:
        tokens = word_tokenize(document)
        tokens = [token.lower() for token in tokens if token.isalpha()]
        tokens = [token for token in tokens if token not in stop_words]
    return processed_data


# In[24]:


#Apply the preprocessing function to the 'text' column
df['processed'] = df['text'].apply(preprocess_article)


# In[26]:


df['processed'] = df['processed'].apply(lambda x: ' '.join(x))  # Append processed document to the list


# In[27]:


pd.set_option('display.max_colwidth', None)
df['processed'][10000]


# # Feature Extraction

# ## Self Preprocessing + TFID

# In[34]:


# SELF - PREPROCESSING + TFID + Training

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


#Feature Extraction using TF-IDF
tfidf_ventorizer = TfidfVectorizer(stop_words='english')
X = tfidf_ventorizer.fit_transform(df['processed'])
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# ## TFID only

# In[11]:


# TFID  + Training
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#Feature Extraction using TF-IDF
tfidf_ventorizer = TfidfVectorizer(stop_words='english')
X = tfidf_ventorizer.fit_transform(df['processed'])

y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# ## CountVectorizer  only

# In[48]:


#Count Vectorizer
# Feature Extraction using Bag of Words
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')  # Initialize the CountVectorizer

#count only
X = vectorizer.fit_transform(df['processed'])  # Fit and transform the text data

#count + self-preprocessing 
#X = vectorizer.fit_transform(df['combined_text'])  # Fit and transform the text data

y = df['category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ## Training

# ### Training using Naive bayes

# In[44]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

#Train the Model (Naive Bayes)
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train, y_train)

y_pred = naive_bayes_classifier.predict(X_test)
y_pred

#model_filename = "nb_countvectorizer_model(30_testsize).pkl" #test size 30

#joblib.dump(naive_bayes_classifier, model_filename)


# ### Training using SVM

# In[ ]:


from sklearn.svm import SVC
# Create an SVM classifier with a linear kernel
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Train the classifier on the training data

# Make predictions on the test data
y_pred = svm_classifier.predict(X_test)

model_filename = "svm_countvectorizer_model(20_testsize).pkl"

#joblib.dump(svm_classifier, model_filename)


# ### Representation

# In[47]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, xticklabels=['predicted_true', 'predicted_fake'], yticklabels=['actual_true', 'actual_fake'],
annot=True, fmt='d', annot_kws={'fontsize':20}, cmap="YlGnBu");



true_neg, false_pos = cm[0]
false_neg, true_pos = cm[1]
accuracy = round((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg),3)
precision = round((true_pos) / (true_pos + false_pos),3)
recall = round((true_pos) / (true_pos + false_neg),3)
f1 = round(2 * (precision * recall) / (precision + recall),3)
print('Accuracy: {}'.format(accuracy))
print('Precision: {}'.format(precision))
print('Recall: {}'.format(recall))
print('F1 Score: {}'.format(f1))

#NBscore = naive_bayes_classifier.score(X_test, y_test)


# In[76]:


#Model Evaluation


y_pred = naive_bayes_classifier.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)


# In[43]:


# Predict unseen data

n = pd.read_csv('predict.csv')
n['processed'] = n['text'].apply(preprocess_text)

#Feature Extraction using TF-IDF
X_new = tfidf_ventorizer.fit_transform(n['processed'])
y_pred = naive_bayes_classifier.predict(X_new)
y_pred


# In[70]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report


print("Test our fake news detection now!")
article = input("Enter your article: ")
fakeortrue = input("Do you think the article is fake (F) or true (T) ? ")

# based on our testing, bow is the best feature, so only bag


print("\nThere are 2 types of training method:")
print("1. SVM")
print("2. Naive Bayes")
training = input("Choose one of the training method: ")

print("\nIt is detecting ...")



# def tfid(article):
#     #Feature Extraction using TF-IDF
#     tfidf_ventorizer = TfidfVectorizer(stop_words='english')
#     X = tfidf_ventorizer.fit_transform(article)

# def countvectorizer(article):
#     vectorizer = CountVectorizer(stop_words='english')  # Initialize the CountVectorizer
#     X = vectorizer.fit_transform(article)  # Fit and transform the text data
    
new_article_vectorized_count = countvectorizer(article)
new_article_vectorized_tfid = tfid(article)
if training == 1:    
    model = SVC()
    model.load('nb_countvectorizer_model(30_testsize).pkl')
else:
#     processed_article = article.apply(preprocess_article) 
    model = MultinomialNB('svm_countvectorizer_model(20_testsize).pkl')
    model.load()
    
prediction = model.predict(new_article_vectorized_count)

if prediction[0] == 1:
    print("The article is predicted as fake.")
else:
    print("The article is predicted as true.")
    
# elif feature == 2:
#     processed_article = article.apply(preprocess_article)
#     countvectorizer(article)
# elif feature == 3:
#     tfid(article)

