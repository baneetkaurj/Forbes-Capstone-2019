# Code for more than 50000 pageviews

import html2text
import re
import pandas as pd
import spacy as sp
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_classif

df = pd.read_csv('2018.csv', nrows=5000)
'''
arts  = list(df['body'])

ts = list(df['timestamp'])
time = []
for x in range(len(arts)):
    t1 = ts[x] 
    time.append(int(datetime.utcfromtimestamp(t1).strftime('%Y%m%d%H%M%S')))


'''

df.shape

# Checking and removing null values
df[df['body'].isnull()]  # No null values
df[df['pageviews'].isnull()]
df.dropna(subset=['pageviews'], inplace=True)
df.dropna(subset=['body'], inplace=True)
df.shape
# Discarding rows with less than 100 words in df['body']
df['body'] = df['body'].astype('str')
df3 = df[df['body'].str.len() > 100]


import numpy as np

# df_mid = (df2[df2['pageviews']<2000])

'''bin_counts, bin_edges, binnumber = stats.binned_statistic(df3['pageviews'], df3['body'], statistic='count', bins=5)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width / 2
print('Bin counts : ', bin_counts)
print('Bin Edges : ', bin_edges)
print('Bin Labels : ', np.unique(binnumber))
'''
body_raw = df3['body']
body_prepro = []
editor_pick = []

for pages in df3['written_by_forbes_staff']: #written_by_forbes_staff  /  editors_pick
    if pages == True:
        editor_pick.append(1)
    else:
        editor_pick.append(0)


'''
bins = np.linspace(min(page_views), max(page_views), 4)
digitized = numpy.digitize(data, bins)
clus = [2000, 50000, 0]
print("The bins are:", clus)
views2 = []


def categorize(l, t2):
    del t2[:]
    for num in l:
        if num <= clus[0]:
            t2.append(int(clus[0]))
        elif num <= bins[2]:
            t2.append(int(clus[1]))
        else:
            t2.append(int(clus[2]))
'''

for line in body_raw:
    line = re.sub(r'font', " ", line)
    line = re.sub(r'donotpaginate', " ", line)
    line = re.sub(r'image', " ", line)
    line = re.sub(r'div', " ", line)
    line = re.sub(r'class', " ", line)
    line = re.sub(r'span', " ", line)
    line = re.sub(r'rsquo', " ", line)
    line = re.sub(r'mdash', " ", line)
    line = re.sub(r'wa', " ", line)
    line = re.sub(r'kelly', " ", line)
    line = re.sub(r'strong', " ", line)
    line = re.sub(r'said', " ", line)
    line = re.sub(r'<a.*>', " LINK ", line)
    line = re.sub(r'<img.*>', " IMAGE ", line)
    line = re.sub(r"\d", " NUMBER ", line)
    line = re.sub(r'\n', " ", line)
    line = re.sub(r'\'s', " ", line)
    line = re.sub(r'U.S.', "United States", line)
    line = re.sub(r'nbsp', " ", line)
    line = re.sub(r'list', " ", line)
    #line = re.sub(r'', "UnitedKingdom", line)
    line = re.sub(r"\W", " ", line)  # removes punctuation marks
    line = re.sub(r"\s+[a-z]\s+", " ", line)  # removes single character
    line = re.sub(r"\s+[a-z][a-z]\s+", " ", line)  # removes double character
    line = re.sub(r"\s+[a-z]$", " ", line)
    line = re.sub(r"^[a-z]\s+", " ", line)
    line = re.sub(r"\s+", " ", line)  # removes extra space
    line = re.sub(r"_", " ", line)  # removes underscore
    line = re.sub(r"n't", " not ", line)
    line = html2text.html2text(line)
    body_prepro.append(line)
# Lemmatization and removing stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, body_prepro):
        return [self.wnl.lemmatize(t) for t in word_tokenize(body_prepro)]


from sklearn.feature_extraction.text import TfidfVectorizer

max_features = 500
vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), max_features=max_features, stop_words=stopwords.words('english'))
y = vectorizer.fit_transform(body_prepro).toarray()
tfidf = vectorizer.get_feature_names()
# indices = np.argsort(vectorizer.idf_)[::-1]
#y = SelectKBest(k=10000).fit_transform(df_t,binnumber)

# top_n = 20


# dividing data into train and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(y, editor_pick, test_size=0.20, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(X_train, y_train)
log_pred = clf.predict(X_test)
accu_log = metrics.accuracy_score(y_test, log_pred)
print(' Logistic Regression Accuracy :', accu_log * 100)

coefs = clf.coef_[0]
top_20 = np.argpartition(coefs, -30)[-30:]  # For positive features
#top_20 = np.argpartition(coefs,30)[:30]  #For negative features

top_20_sorted = top_20[np.argsort(coefs[top_20])]
top_20_desc = top_20_sorted[::-1]
for i in top_20_desc:
    print(int(coefs[i]*100), tfidf[i])


d = {}
for a, x in zip(coefs, tfidf):
    d[x] = a

import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(font_path='Times New Roman.ttf', background_color='white', width=2400, height=1800)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.tight_layout(pad=0)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()