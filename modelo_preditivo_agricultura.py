#importando as bibliotecas para usar
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# alocando a base de dados em uma variavel 
crops = pd.read_csv("soil_measures.csv")

#checar os valores ausentes
crops.isna().sum()

#verificar quantos crops temos, avaliando valores com mais de uma classe
crops["crop"].unique()

#dividir os dados em x e y
X = crops.drop(columns="crop")
y = crops["crop"]

#dividr os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#criando um dicionário para armazenar a performance de cada modelo
feature_performance = {}

#Treinar um modelo de regressão linear para cada feature
for feature in ["N", "P", "K", "ph"]:
    log_reg = LogisticRegression(multi_class="multinomial", solver='lbfgs', max_iter=200)
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])
    
    #Calcular o score de F1, a média harmonica da precisão e do recall
    f1 = metrics.f1_score(y_test, y_pred, average="weighted") 
    
    #adicionando as pontuações pares feature-f1 ao dicionario
    feature_performance[feature] = f1
    print(f"F1-score for {feature}: {f1}")

#Guardar em best_predictive_feature no dicionário
best_predictive_feature = {"K": feature_performance["K"]}
best_predictive_feature