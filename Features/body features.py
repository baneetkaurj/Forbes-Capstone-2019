import html2text
import re
import pandas as pd
import spacy as sp
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def create_bins():
    print("\n\nGenerating bins...")
    for pages in df['pageviews']:  # written_by_forbes_staff
        pages = int(pages)
        if pages <= 2000:
            pv.append('low')
        elif pages <= 50000 | pages > 2000:
            pv.append('med')
        elif pages > 50000:
            pv.append('high')
    print("\n\nBins generated.")


def pre_process():
    print("\n\nPreprocessing the data...")
    df.dropna(subset=['pageviews'], inplace=True)
    df.dropna(subset=['body'], inplace=True)
    print(df.shape)
    df['body'] = df['body'].astype('str')
    df[df['body'].str.len() > 100]
    body_raw = df['headline']  # headline , body
    for line in body_raw:
        line = re.sub(r'[?]', " h_question ", line)
        line = re.sub(r'[!]', " h_exclamation ", line)
        line = re.sub(r'[:]', " h_colon ", line)
        line = re.sub(r'[%]', " h_percent ", line)
        line = re.sub(r'[0-9]+', " h:<int> ", line)
        line = re.sub(r'[Bb]attle [Rr]oyal', " BattleRoyale ", line)
        line = re.sub(r'[Mm]odel [Xx]', " ModelX ", line)
        line = re.sub(r'[Aa]rtificial [Ii]ntelligence', " artificial_intelligence ", line)
        line = re.sub(r'[Aa][Ii]', " artificial_intelligence ", line)
        body_prepro.append(line)
    print("\n\nData preprocessed.")
    create_bins()


def model():
    print("\n\nGenerating the logistic regression model...")
    class LemmaTokenizer(object):
        def __init__(self):
            self.wnl = WordNetLemmatizer()

        def __call__(self, body_prepro):
            return [self.wnl.lemmatize(t) for t in word_tokenize(body_prepro)]

    max_features = 500
    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), max_features=max_features,
                                 stop_words=stopwords.words('english'))
    y = vectorizer.fit_transform(body_prepro).toarray()
    tfidf = vectorizer.get_feature_names()
    indices = np.argsort(vectorizer.idf_)[::-1]
    X_train, X_test, y_train, y_test = train_test_split(y, pv, test_size=0.20, random_state=0)
    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(X_train, y_train)
    log_pred = clf.predict(X_test)
    accu_log = metrics.accuracy_score(y_test, log_pred)
    print('\n\nLogistic Regression Accuracy :', accu_log * 100)
    coefs = clf.coef_[0]
    top_20 = np.argpartition(coefs, -30)[-30:]  # For positive features
    # top_20 = np.argpartition(coefs,30)[:30]  #For negative features
    top_20_sorted = top_20[np.argsort(coefs[top_20])]
    top_20_desc = top_20_sorted[::-1]
    for i in top_20_desc:
        print(int(coefs[i]*100), tfidf[i])

    for a, x in zip(coefs, tfidf):
        d[x] = a
    print("\n\nModel fitted.")


def create_wordcloud():
    print("\n\nGenerating wordcloud...")
    wordcloud = WordCloud(background_color='white', width=800, height=400)
    wordcloud.generate_from_frequencies(frequencies=d)
    plt.figure(figsize=(20, 10), facecolor='k')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    print("Reading the data...")
    df = pd.read_csv('2018.csv', nrows=10000)
    body_prepro = []
    pv = []
    d = {}
    pre_process()
    # create_bins()
    model()
    create_wordcloud()
