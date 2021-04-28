import pandas as pd
import numpy as np
import re
import nltk
nltk.download()
import string
from sklearn.linear_model import LogisticRegression

### Read training file and testing file
df_train = pd.read_csv("Corona_NLP_train.csv", encoding='ISO-8859-1')
df_test = pd.read_csv("Corona_NLP_test.csv", encoding='ISO-8859-1')
print(df_train.shape)
print(df_test.shape)

### Narrow our classification to three classes: Positive, Neutral, Negative and drop the unnecessary column
df_train = df_train.replace(regex={'Extremely Positive': 'Positive', 'Extremely Negative': 'Negative'})
print(pd.unique(df_train['Sentiment']))
df_train = df_train.drop(['UserName', 'ScreenName', 'Location', 'TweetAt'], axis=1)

df_test = df_test.replace(regex={'Extremely Positive': 'Positive', 'Extremely Negative': 'Negative'})
print(pd.unique(df_test['Sentiment']))
df_test = df_test.drop(['UserName', 'ScreenName', 'Location', 'TweetAt'], axis=1)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


### Text Cleaning
def clean_text(df):
    all_reviews = list()
    lines = df["OriginalTweet"].values.tolist()
    for text in lines:
        # Lower Casing
        text = text.lower()

        # Remove numbers, URL and punctuation
        text = re.sub(r'\d+', '', text)
        pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = pattern.sub('', text)

        tokens = word_tokenize(text)
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        stop_words.discard("not")
        words = [w for w in words if not w in stop_words]
        words = ' '.join(words)

        # lemmatize string
        lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(text)
        text = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]

        all_reviews.append(words)

    return all_reviews


###Split to testing set and development set
from sklearn.model_selection import train_test_split

df_x = clean_text(df_train)
y = df_train['Sentiment'].tolist()
x_train, x_dev, y_train, y_dev = train_test_split(df_x, y, test_size=0.10, random_state=42)

x_test = clean_text(df_test)
y_test = df_test['Sentiment'].tolist()

### Extracting features from text files using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
x_train_vec = vectorizer.fit_transform(x_train).toarray()
x_dev_vec = vectorizer.transform(x_dev).toarray()

###Using LinearSVC model for predictions
from sklearn.svm import LinearSVC
from sklearn import metrics

clf = LinearSVC()
clf.fit(x_train_vec, y_train)
prediction = clf.predict(x_dev_vec)
print("The prediction accuracy of the development set with LinearSVC model is:")
print(metrics.accuracy_score(prediction, y_dev))

## Logistic Regression
model_LR = LogisticRegression()
model_LR.fit(x_train, y_train)

y_predict_LR = model_LR.predict(x_test)
model_LR_score = model_LR.score(x_test, y_test)

print("-------Logistic Regression-------")
print("Accuracy: ", model_LR_score*100)
