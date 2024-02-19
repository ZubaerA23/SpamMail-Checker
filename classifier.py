import pandas as pd
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


emails = [[0, 'Hello, How are you'],
          [1, 'Win epic prices today, win from home'],
          [0, 'Call me now. I need to speak with you'],
          [0, 'Can we have a meeting at your place tomorrow?']]

df = pd.DataFrame(np.array(emails), columns = ['tag', 'mail_message'])

print(df.head())

X_train , X_test ,y_train, y_test = train_test_split(df['mail_message'], df['tag'], random_state = 1)

print("Number of rows in total set: {}" .format(df.shape[0]))
print("Number of rows in training set: {}" .format(X_train.shape[0]))
print("Number of rows in test set: {}" .format(X_test.shape[0]))


count_vector = CountVectorizer()

training_data = count_vector.fit_transform(X_train)
print(training_data)
print(count_vector.get_feature_names())

testing_data = count_vector.transform(X_test)

naive_bayes = MultinomiaNB()
naive_bayes.fit(training_data , y_train)

predictions = naive_bayes.predict(testing_data)
print('the prediction was: ' , predictions)
print('Accurance score: ', format(accuracy_score(y_test, predictions)))