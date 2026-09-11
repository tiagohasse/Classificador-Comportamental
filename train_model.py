import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. Carrega o Extrovert vs. Introvert Behavior Dataset localmente
df = pd.read_csv('personality_dataset.csv')

# 2. Tratamento mínimo de dados
df_clean = df

# Features e target binárias: 'Yes' vira 1 (Sim para medo de palco, ou sim drenado depois de socializar, ou Extrovertido), 'No' vira 0 (Não para medo de palco, ou não drenado depois de socializar, ou Introvertido)
df_clean['Stage_fear'] = (df_clean['Stage_fear'] == 'Yes').astype(int)
df_clean['Drained_after_socializing'] = (df_clean['Drained_after_socializing'] == 'Yes').astype(int)
df_clean['Extrovert'] = (df_clean['Personality'] == 'Extrovert').astype(int)

features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']
X = df_clean[features] #features/caracteristicas/atributos são as variáveis independentes
y = df_clean['Extrovert'] # target/label/rótulo/classe é a variável dependente

# 3. Divisão Treino e Teste
# evita não decorar respostas
# random_state -> determinístico
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 4. Treinamento do Modelo
loss_svc = 'squared_hinge'
penalty_svc = 'l1'

model_lsvc_b = LinearSVC(loss=loss_svc, penalty=penalty_svc, max_iter=100, random_state=42)
model_lsvc_b.fit(X_train, y_train)

acc = accuracy_score(y_test, model_lsvc_b.predict(X_test))
print(f"[4/5] SVC B treinada! Acurácia de teste: {acc * 100:.2f}%")

# 5. Exportação do Artefato
joblib.dump(model_lsvc_b, 'personality_model.pkl')
print("[5/5] Modelo salvo com sucesso em 'personality_model.pkl'!")