import pandas as pd

from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_classif

# all_files = ['2010.csv', '2011.csv', '2012.csv', '2013.csv', '2014.csv', '2015.csv']
#
# li = []
# dfl = []
# for filename in all_files:
#     li = pd.read_csv(filename, index_col=None, header=0, low_memory=False, encoding='utf-8')
#     dfl.append(li)

# df = pd.concat(li, axis=0, ignore_index=True, sort=False)
# df = pd.read_csv('2010.csv', encoding='utf-8')
df = pd.read_csv('2018.csv')
# df = pd.DataFrame(dfl)
df.shape

# Checking and removing null values
df[df['body'].isnull()]  # No null values
df[df['pageviews'].isnull()]
df[df['timestamp'].isnull()]  # No null values
df.dropna(inplace = True)
# df.dropna(subset=['pageviews'], inplace=True)
# df.dropna(subset=['body'], inplace=True)
num_col = ['pageviews', 'timestamp']
# df[num_col] = pd.to_numeric(df[num_col], errors='coerce')
# df = df.dropna(subset=[num_col])
# df[num_col] = df[num_col].astype(int)
print(df.shape)
# Discarding rows with less than 100 words in df['body']
df['body'] = df['body'].astype('str')
body = df['body']
import numpy as np

timestamp = np.array(df['timestamp'])

month = []
day = []
hours = []
week = []
count = 0

for x in range(len(body)):
    count = count + 1
    t1 = timestamp[x]
    t1 = int(t1)
    t1 = int(t1 / 1000)
    month.append("Month_" + str(datetime.utcfromtimestamp(t1).strftime('%m')) + " ")
    day.append("Day_" + str(datetime.utcfromtimestamp(t1).strftime('%d')) + " ")
    hours.append("Hour_" + str(datetime.utcfromtimestamp(t1).strftime('%H')) + " ")
    week.append("Week_" + str(datetime.utcfromtimestamp(t1).strftime('%W')) + " ")


body_raw = df['body']
body_prepro = []
pv = []


for pages in df['pageviews']: #written_by_forbes_staff
    print(pages)
    if pages <= 2000:
        pv.append('low')
    elif pages <= 50000 | pages > 2000:
        pv.append('med')
    elif pages > 50000:
        pv.append('high')



# Lemmatization and removing stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, body_prepro):
        return [self.wnl.lemmatize(t) for t in word_tokenize(body_prepro)]


X_svd = np.core.defchararray.add(month, day)
X_svd = np.core.defchararray.add(X_svd, hours)
X_svd = np.core.defchararray.add(X_svd, week)


#X_svd = np.core.defchararray.add(X_svd, day)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
max_features = 500
vectorizer = TfidfVectorizer(max_features=max_features)
y = vectorizer.fit_transform(X_svd).toarray()
tfidf = vectorizer.get_feature_names()
indices = np.argsort(vectorizer.idf_)[::-1]


top_n = 20



# dividing data into train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(y, pv, test_size=0.20, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

clf = LogisticRegression(solver='lbfgs').fit(X_train, y_train)
log_pred = clf.predict(X_test)
accu_log = metrics.accuracy_score(y_test, log_pred)
print(' Logistic Regression Accuracy :', accu_log * 100)

coefs = clf.coef_[0]
top_20 = np.argpartition(coefs, -20)[-20:]  # For positive features
#top_20 = np.argpartition(coefs,30)[:30]  #For negative features

top_20_sorted = top_20[np.argsort(coefs[top_20])]
top_20_desc = top_20_sorted[::-1]
for i in top_20_desc:
    print(coefs[i], '\t', tfidf[i])


d = {}
for a, x in zip(coefs, tfidf):
    d[x] = a

import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(background_color='white', width=2400, height=1800)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.tight_layout(pad=0)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()