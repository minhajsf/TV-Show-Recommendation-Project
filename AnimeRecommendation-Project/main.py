from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
import http.client
import json
import requests 
from mal import AnimeSearch
anime = AnimeSearch("CowBoy Bepop")

goodAnimeList = [] 
goodLength = 5
while goodLength > 0:
    anime = input("Enter your favorite Anime: ")
    goodAnimeList.append(anime)
    goodLength -= 1

badAnimeList = [] 
badLength = 5
while badLength > 0:
    anime = input("Enter your least favorite Anime: ")
    badAnimeList.append(anime)
    badLength -= 1

testAnimeList= [] 
testLength = 5
while testLength > 0:
    anime = input("Enter Anime you haven't watched: ")
    testAnimeList.append(anime)
    testLength -= 1
    
 
good_overview = [ ]
bad_overview = [ ]
test_overview = [ ]# call in test data sets

#putting in the overviews
for title in goodAnimeList:
  overview = AnimeSearch(title)
  good_overview.append(overview.results[0].synopsis)
for title in badAnimeList:
  overview = AnimeSearch(title)
  bad_overview.append(overview.results[0].synopsis)
for title in testAnimeList:
  overview = AnimeSearch(title)
  test_overview.append(overview.results[0].synopsis)

# combine our positive and negative texts 
# into a  training set to put into the classifier 
training_texts = good_overview + bad_overview


# tell the machine that the first five are posititve and the next ones are negative. 
# matches indices then classifies the anime
training_labels = ["good"] * len(good_overview) + ["bad"] * len(bad_overview)

#setting up the vectorizer which is the first component of machine learning
vectorizer = CountVectorizer()

vectorizer.fit(training_texts)

# transform all of our training texts into vector form 
training_vectors = vectorizer.transform(training_texts)


# the same to our test text, each of these is a list of numbers 
test_texts = test_overview 
testing_vectors = vectorizer.transform(test_texts)

# we create our classifier and train it  by "showing" the training texts and the associated labels 
#It iterates over the data a few times,  trying different rules, until it finds a set of rules that works 
classifier = tree.DecisionTreeClassifier()
classifier.fit(training_vectors, training_labels)


# now we ask the computer to guess whether our test texts  
# are more similar to the positive texts or the negative ones
Results = classifier.predict(testing_vectors)
for i in range(0, len(testAnimeList)):    
    print("According to the program "+testAnimeList[i]+" would be " +Results[i]),

    
# export the model to a file so that we can visualise it
# copy the content from `tree.dot` to http://www.webgraphviz.com/ to 
# see what the tree looks like
tree.export_graphviz(
    classifier,
    out_file='tree.dot',
    feature_names=vectorizer.get_feature_names(),
    class_names=["bad","good"]
    
) 




