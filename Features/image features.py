import os
from google.cloud import vision
import pandas as pd
import io
import re
import requests
from bs4 import BeautifulSoup
import urllib.request
import numpy as np

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "My First Project-6ade6d21fa6c.json"
df = pd.read_csv('2018.csv', nrows=50, skiprows=(1, 3000))
df[df['url'].isnull()]
client = vision.ImageAnnotatorClient()
image_features = []
image_features_particular = []
count = 0
for url1 in df['url']:
    image_features_particular = []
    count = count + 1
    print(count)
    response = requests.get(url1)
    url = BeautifulSoup(response.text, 'html.parser')
    try:
        filename = re.search(r'<progressive-image.+progressive-image>', str(url)).group(0)
        img_url = re.search(r'https.+?(jpg|jpeg|png)', filename).group(0)
        urllib.request.urlretrieve(img_url, "temp.jpg")
    except:
        count = count - 1
        df.drop(df.index[count], inplace=True)
        continue
    path = 'temp.jpg'
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    if not labels:
        count = count - 1
        df.drop(df.index[count], inplace=True)
        continue
    for label in labels:
        label.description = re.sub(r'label', " ", label.description)
        label_tag = "image_" + label.description
        image_features_particular.append(label_tag)
        print(label_tag)
    image_features_particular = str(image_features_particular)
    image_features.append(image_features_particular)
    print("\n")

pv = []

for pages in df['pageviews']:
    if pages < 2000:
        pv.append("low")
    elif pages > 50000:
        pv.append("high")
    else:
        pv.append("med")

from sklearn.feature_extraction.text import TfidfVectorizer
max_features = 500
vectorizer = TfidfVectorizer(max_features=max_features)
y = vectorizer.fit_transform(image_features).toarray()
tfidf = vectorizer.get_feature_names()
indices = np.argsort(vectorizer.idf_)[::-1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(y, pv, test_size=0.30, random_state=0)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics

clf = LogisticRegression(solver='lbfgs', multi_class='auto').fit(X_train, y_train)
log_pred = clf.predict(X_test)
accu_log = metrics.accuracy_score(y_test, log_pred)
print(' Logistic Regression Accuracy :', accu_log * 100)
coefs = clf.coef_[0]
top_20 = np.argpartition(coefs, -20)[-20:]  # For positive features
# top_20 = np.argpartition(coefs,30)[:30]  #For negative features

top_20_sorted = top_20[np.argsort(coefs[top_20])]
top_20_desc = top_20_sorted[::-1]
for i in top_20_desc:
    print(int(coefs[i] * 100), tfidf[i])
