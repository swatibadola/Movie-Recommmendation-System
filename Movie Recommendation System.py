#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#DATA COLLECTION - Loading the csv file
movies_data = pd.read_csv('C:/Users/DELL/OneDrive/Desktop/movies.csv')

#Printing the first 5 rows of the dataframe
movies_data.head()

# Selecting the relevant features for recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
print(selected_features)

# Replacing the null values with null string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

#Combining all the 5 selected features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
print(combined_features)

#Converting data to feature vector
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
print(feature_vectors)

#Cosine Similarity
similarity = cosine_similarity(feature_vectors)
print(similarity)

#Getting the movie name from the user
movie_name = input('Enter your favourite movie name: ')

#Creating a list with all the movie names given in the dataset
list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)

#Finding the close match for the movie name given by the user
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)

close_match = find_close_match[0]
print(close_match)

#Finding the index of the movie with title 
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)

#Getting a list of the similar movies
similarity_score = list(enumerate(similarity[index_of_the_movie]))

#Sorting the movie based on their similarity score
sorted_similar_movies = sorted(similarity_score, key=lambda x:x[1], reverse=True)
print(sorted_similar_movies)

#Print the name of similar movies based on the index
print('Movies suggested for you: \n')
i=1
for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if(i<40):
        print(i, '.', title_from_index)
        i+=1
        
