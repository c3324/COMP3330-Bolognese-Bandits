import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

trainDataSet = pd.read_json("./tweet_topic_single/dataset/split_coling2022_temporal/train_2020.single.json", lines=True)
testDataSet = pd.read_json("./tweet_topic_single/dataset/split_coling2022_temporal/train_2020.single.json", lines=True)

trainInputs = trainDataSet.text
trainTargets = trainDataSet.label

testInputs = testDataSet.text
testTargets = testDataSet.label

cv = CountVectorizer()
trainCountVector = cv.fit_transform(trainInputs.values)

model = MultinomialNB()
model.fit(trainCountVector, trainTargets)

testInputsVector = cv.transform(testInputs)

pred = model.predict(trainCountVector)

print(classification_report(testTargets, pred))
