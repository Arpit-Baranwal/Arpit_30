import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
messages = pd.read_csv(r"F:\studies\NLP\spam classifier\spam.csv",sep=',', names=["label","message"])
ls = WordNetLemmatizer()
corpus = []

for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', str(messages['message'][i]))
    review = review.lower()
    review =  review.split()

    review = [ls.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
x = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test =  train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB()
spam_detect_model.fit(X_train,Y_train)

y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix

confusion_m = confusion_matrix(Y_test,y_pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test,y_pred)
accuracy = accuracy*100
print("Your accuracy is " + str(accuracy) + "%")