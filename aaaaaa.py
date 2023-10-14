"""import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Carregar sua base de dados (substitua 'movies_metadata.csv' pelo nome do seu arquivo)
data = pd.read_csv('movies_metadata.csv', low_memory=False, nrows=8000)

# Preencher valores nulos
data['overview'] = data['overview'].fillna('')
data['title'] = data['title'].fillna('')

# Escolher os atributos para treinamento
X = data[['title', 'overview']]

# Inicializar o vetorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Aplicar o vetorizador aos atributos de treinamento
X_tfidf = tfidf_vectorizer.fit_transform(X['title'] + ' ' + X['overview'])

# Inicializar o modelo Naive Bayes (MultinomialNB)
model = MultinomialNB()

# Avaliações reais dos filmes (suponha que você tenha uma coluna no DataFrame chamada 'vote_average')
y = data['vote_average'] >= 4.0  # Classificar como 'recommended' se a avaliação for >= 4.0

# Treinar o modelo com todos os dados
model.fit(X_tfidf, y)

# Entrada do usuário
movie_title = input("Digite o título do filme: ")
movie_overview = input("Digite a sinopse do filme: ")

# Vetorizar a entrada do usuário
user_input_tfidf = tfidf_vectorizer.transform([movie_title + ' ' + movie_overview])

# Fazer previsões para a entrada do usuário
user_prediction = model.predict(user_input_tfidf)

# Exibir a classificação do filme
if user_prediction[0]:
    print("Classificação do Filme: Recommended")
else:
    print("Classificação do Filme: Not Recommended")
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Carregar sua base de dados (substitua 'movies_metadata.csv' pelo nome do seu arquivo)
data = pd.read_csv('movies_metadata.csv', low_memory=False, nrows=8000)

# Preencher valores nulos
data['overview'] = data['overview'].fillna('')
data['title'] = data['title'].fillna('')
data['popularity'] = data['popularity'].fillna('')


# Escolher os atributos para treinamento
X = data[['title', 'overview']]

# Criar atributos-alvo com base nas avaliações dos filmes
y = data['popularity']

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o vetorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Aplicar o vetorizador aos atributos de treinamento
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['title'] + ' ' + X_train['overview'])
X_test_tfidf = tfidf_vectorizer.transform(X_test['title'] + ' ' + X_test['overview'])

# Inicializar o modelo Naive Bayes (MultinomialNB)
model = MultinomialNB()

# Treinar o modelo
model.fit(X_train_tfidf, y_train)

# Fazer previsões
predictions = model.predict(X_test_tfidf)

print(f'Predict: {predictions}')

r = 0
for i in predictions:
    if i == 'not recommended':
        r+=1
print(r)
print(len(predictions))

print()
print(f'Overview: {data["overview"]}')
#print(f'x_train_tfidf: {X_train_tfidf}')
#print(f'x_test_tfidf: {X_test_tfidf}')
print(data['title'], data['vote_average'])

print()
print(data['genres'][0])
print(data['popularity'])
print(data['title'])

# Avaliar a precisão do modelo
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Precisão do modelo:", accuracy)


"""import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Carregar sua base de dados (substitua 'movies_metadata.csv' pelo nome do seu arquivo)
data = pd.read_csv('movies_metadata.csv', low_memory=False, nrows=8000)

# Preencher valores nulos
data['overview'] = data['overview'].fillna('')
data['title'] = data['title'].fillna('')

# Escolher os atributos para treinamento
X = data[['title', 'overview']]

# Criar atributos-alvo com base nas avaliações dos filmes
y = ['recommended' if vote_average >= 8 else 'not recommended' for vote_average in data['vote_average']]

# Inicializar o vetorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Aplicar o vetorizador aos atributos
X_tfidf = tfidf_vectorizer.fit_transform(X['title'] + ' ' + X['overview'])

# Inicializar o modelo Naive Bayes (MultinomialNB)
model = MultinomialNB()

# Treinar o modelo
model.fit(X_tfidf, y)



# Função para encontrar filmes semelhantes
def find_similar_movies(movie_description, num_movies=5):
    movie_tfidf = tfidf_vectorizer.transform([movie_description])
    prediction = model.predict(movie_tfidf)

    # Filtrar os filmes recomendados
    if prediction[0] == 'recommended':
        similar_movies = data[data['vote_average'] >= 8]
    else:
        similar_movies = data[data['vote_average'] < 8]

    similar_movies = similar_movies[similar_movies['title'] != movie_description]
    similar_movies = similar_movies.sample(num_movies)

    return similar_movies['title']

# Exemplo de uso da função
input_movie = input("Digite o título de um filme: ")
similar_movies = find_similar_movies(input_movie)

if len(similar_movies) > 0:
    print(f"Filmes semelhantes a '{input_movie}':")
    for movie in similar_movies:
        print(movie)

    # Avaliar a precisão do sistema
    accuracy = accuracy_score([input_movie] * len(similar_movies), similar_movies)
    print("Precisão do sistema de recomendação:", accuracy)
else:
    print(f"Não foram encontrados filmes semelhantes a '{input_movie}'.")
"""