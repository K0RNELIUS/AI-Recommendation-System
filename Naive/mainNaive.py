import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Carregar sua base de dados (substitua 'movies_metadata.csv' pelo nome do seu arquivo)
data = pd.read_csv('../movies_metadata.csv', low_memory=False, nrows=10000)

# Preencher valores nulos
data['overview'] = data['overview'].fillna('')
data['title'] = data['title'].fillna('')

# Escolher os atributos para treinamento
X = data[['title', 'overview']]

# Criar atributos-alvo com base nas avaliações dos filmes
y = ['recommended' if vote_average >= 5 else 'not recommended' for vote_average in data['vote_average']]

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

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

# Avaliar a precisão do modelo

accuracy = accuracy_score(y_test, predictions)
print("Precisão do modelo:", accuracy)


# Classification Report
cr = classification_report(y_test, predictions)
print("=============================")
print("Classification Report:")
print("=============================")
print(cr)
print("=============================")
