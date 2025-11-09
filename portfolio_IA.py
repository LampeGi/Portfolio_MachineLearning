import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

#1) Carregar os dados do Excel
dados = pd.read_excel('Dataset - Portfólio Python com IA.xlsx')

#2) Tirar linhas que não têm a coluna 'Satisfação' preenchida
dados = dados.dropna(subset=['Satisfação']).copy()

#3) Se tiver a coluna "Nome do cliente" no dataset, remover porque não influenciará no modelo
if 'Nome do cliente' in dados.columns:
    dados = dados.drop(columns=["Nome do cliente"])

#4) Transformar os dados de texto em números para o modelo entender
tipos_de_comida = {"Carne": 1, "Frango": 2, "Lasanha": 3}
atendimento_do_garcom = {"Sim": 1, "Não": 2}
satisfacao = {"Satisfeito": 1, "Insatisfeito":0}

#5) Aplicar a transformação nas colunas
dados['Prato'] = dados['Prato'].replace(tipos_de_comida).infer_objects(copy=False).astype(float)
dados['Atendimento'] = dados['Atendimento'].replace(atendimento_do_garcom).infer_objects(copy=False).astype(float)
dados['Satisfação'] = dados['Satisfação'].replace(satisfacao).infer_objects(copy=False).astype(float)

#6) Separar os dados em “inputs” (o que o modelo usa) e “target” (o que queremos prever)
X = dados[['Prato', 'Gasto', 'Atendimento']] # o que o modelo irá usar
y = dados['Satisfação'] # o que o modelo precisa prever

#7) Dividir os dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#8) Criar e treinar o modelo de Random Forest
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

#9) Testar como o modelo se saiu: ver acurácia, precisão e a matriz de confusão
y_pred = modelo.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
precision_score = precision_score(y_test, y_pred)
matriz = confusion_matrix(y_test, y_pred)

print(f"Acurácia do modelo: {acuracia}")
print(f"Precisão do modelo: {precision_score}")
print("\nMatriz de confusão: [TN FP] [FN TP]")
print(matriz)

# 10) Testar o modelo utilizando todas as opções de pratos 
todos_os_pratos = [
    {'Prato':'Frango', 'Gasto': 100, 'Atendimento': 'Não'},
    {'Prato':'Lasanha', 'Gasto': 100, 'Atendimento': 'Não'},
    {'Prato':'Carne', 'Gasto': 100, 'Atendimento': 'Não'}
]

for cliente in todos_os_pratos:
    novo_enc = pd.DataFrame([[ 
        tipos_de_comida[cliente['Prato']],
        cliente['Gasto'],
        atendimento_do_garcom[cliente['Atendimento']]
    ]], columns=X.columns)

# 11) Prever e mostrar satisfação dos clientes para cada prato
    pred = modelo.predict(novo_enc)[0]
    resultado = 'Satisfeito' if pred == 1 else 'Insatisfeito'
    print(f"Para o prato {cliente['Prato']}, o modelo prevê: {resultado}")
    

