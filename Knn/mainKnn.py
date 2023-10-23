import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from ast import literal_eval
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report

metadata = pd.read_csv('../movies_metadata.csv', low_memory=False)

credits = pd.read_csv('../credits.csv')
keywords = pd.read_csv('../keywords.csv')

# Remove linhas com IDs ruins
metadata = metadata.drop([19730, 29503, 35587])

# Conversão dos IDs
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

# Unindo keyword e credits na base de dados principal
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

# Itera nos parâmetros relevantes para o algoritmo
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

# Define functions to extract director, cast, genres, and keywords
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

# Define new director, cast, genres, and keywords features
metadata['director'] = metadata['crew'].apply(get_director)
features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

# Função que remove espaços e deixa o conglomerado de informações minúsculo
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Aplica a padronização de limpeza dos dados à todos parâmetros
features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

# Une todos textos relevantes
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
metadata['soup'] = metadata.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])

knn_model = NearestNeighbors(n_neighbors=11, algorithm='auto', metric='cosine')
knn_model.fit(count_matrix)

C = metadata['vote_average'].mean()

# Calculando a quantidade mínima de votos necessárias
m = metadata['vote_count'].quantile(0.70)

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)

# Aplicando a função da média ponderada para calcular o "score" de todos os filmes
metadata['weighted_rating'] = metadata.apply(weighted_rating, axis=1)

# Definindo o limite da popularidade baseando-se no "score" obtido
popularity_threshold = metadata['weighted_rating'].quantile(0.60)

# Função para validar a superioridade do score
def get_ground_truth(title):
    movie_idx = metadata.index[metadata['title'] == title].tolist()[0]
    return metadata['weighted_rating'][movie_idx] > popularity_threshold

# Função que retorna a validação para cada titulo de recomendação obtido
def get_ground_truth_labels(recommendations):
    return [get_ground_truth(movie) for movie, _ in recommendations]

def get_knn_recommendations(title):
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(metadata['soup'])

    knn_model = NearestNeighbors(n_neighbors=11, algorithm='auto', metric='cosine')
    knn_model.fit(count_matrix)

    movie_idx = metadata.index[metadata['title'] == title].tolist()[0]

    # Processa e vetoriza informações do título recebido
    movie_vector = count.transform([metadata['soup'][movie_idx]])

    # Encontra os vizinhos próximos
    _, indices = knn_model.kneighbors(movie_vector, n_neighbors=11)

    # Especifica os índices dos vizinnhos
    movie_indices = indices.flatten()

    # Cria a lista de filmes recomendados com base nos índices obtidos
    recommendations = [(metadata['title'].iloc[i], i) for i in movie_indices if i != movie_idx]

    return recommendations

recommendations = get_knn_recommendations("Star Wars")

print("=============================")
print("Recommended Movies using Knn:")
print("=============================")
for movie, index in recommendations:
    print(f"{index}\t{movie}")

ground_truth_labels = get_ground_truth_labels(recommendations)

predicted_labels = [1] * len(recommendations)

cr = classification_report(ground_truth_labels, predicted_labels)

print("=============================")
print("Classification Report:")
print("=============================")
print(cr)
print("=============================")