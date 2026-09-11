# Classificador Comportamental (Extrovertido vs. Introvertido)

Este projeto é uma aplicação de Machine Learning supervisionado de ponta a ponta para **Classificação de Personalidade**. 

O objetivo do projeto é receber hábitos do cotidiano e preferências sociais de uma pessoa e prever diretamente a sua classe binária: **Extrovertido** ou **Introvertido**. A solução conta com treinamento de modelo preditivo em Scikit-Learn, disponibilização de API REST via FastAPI e uma interface web para interação com o usuário.

---

## Proposta do Projeto & Respostas ao Questionário

### 1. Qual dataset foi escolhido e qual problema ele representa?
* **Dataset:** *Extrovert vs. Introvert Behavior Dataset* (`personality_dataset.csv`), contendo 2.900 registros tabulados.
* **Problema:** Questionários de personalidade comuns costumam ser longos e exaustivos. O problema abordado é classificar o perfil comportamental de um indivíduo com base em um conjunto enxuto de 7 hábitos diários e reações a interações sociais.

### 2. Qual é a variável-alvo (target) que será prevista?
* A variável-alvo é **`Personality`**, tratada como a variável binária **`Extrovert`** (onde `1 = Extrovert` e `0 = Introvert`).

### 3. Quais são as classes possíveis?
* **`Extrovertido` (Extrovert):** Indivíduo com comportamento orientado para fora, dinâmico em grupos e com facilidade de exposição pública.
* **`Introvertido` (Introvert):** Indivíduo com comportamento voltado para a reflexão, que recarrega energias na solitude e prefere círculos menores.

### 4. Quais informações são utilizadas como entrada do modelo?
São utilizadas 7 características como variáveis independentes:
1. **`Time_spent_Alone`**: Horas por dia passadas sozinho(a) (numérico).
2. **`Stage_fear`**: Medo de palco ou desconforto ao falar em público (0 = Não, 1 = Sim).
3. **`Social_event_attendance`**: Frequência em festas e eventos sociais (escala de 0 a 10).
4. **`Going_outside`**: Dias por semana em que sai de casa para passeios ou lazer (0 a 7 dias).
5. **`Drained_after_socializing`**: Sensação de esgotamento após socializar (0 = Não, 1 = Sim).
6. **`Friends_circle_size`**: Quantidade de amigos próximos (numérico).
7. **`Post_frequency`**: Frequência de postagens em redes sociais (escala de 0 a 10).

### 5. Quem utilizaria essa aplicação e com qual finalidade?
* **Equipes de RH / Recrutamento:** Para triagem ágil e auxílio na alocação de candidatos em perfis de equipe que demandem maior foco individual ou trabalho constante de comunicação.
* **Gestores de Equipe:** Para entender as preferências de trabalho de seus liderados (como preferência por tarefas individuais ou reuniões em grupo).
* **Usuários Finais:** Para fins de autoconhecimento de forma rápida e simples.

### 6. O que a aplicação fará com a classificação produzida pelo modelo?
A aplicação processa as 7 entradas através do modelo treinado (`personality_model.pkl`), identifica a classe (**Extrovertido** ou **Introvertido**) e calcula as **porcentagens de probabilidade** para cada perfil comportamental, informando ao usuário o resultado e o nível de certeza.

### 7. Como seria a interface ou experiência de uso dessa solução?
A interface ([`index.html`](index.html)) utiliza controles interativos de incremento e decremento (*steppers* com botões `+` e `−`) e seletores binários:
* Permite ajuste fino e específico de cada métrica (como 4.5 horas, 6 amigos ou 3 dias), sem a necessidade de sliders ou digitação manual de números. Suporta clique simples ou pressão contínua para alteração rápida.
* Possui um botão "Classificar Personalidade" que envia os dados formatados para a API.
* Exibe na tela o cartão de resultado com a classificação final em destaque (**Extrovertido** ou **Introvertido**), uma barra visual comparativa e as porcentagens exatas de probabilidade.

---

## Tecnologias Utilizadas

* **Machine Learning & Manipulação de Dados:**
  * [Python 3](https://www.python.org/)
  * [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
  * [Scikit-Learn](https://scikit-learn.org/)
  * [Joblib](https://joblib.readthedocs.io/)

* **Backend / API:**
  * [FastAPI](https://fastapi.tiangolo.com/)
  * [Pydantic](https://docs.pydantic.dev/)
  * [Uvicorn](https://www.uvicorn.org/)

* **Frontend:**
  * HTML5, CSS3 e JavaScript Vanilla (Fetch API)

---

## Estrutura de Arquivos

```text
Classificador-Comportamental/
├── app.py                      # API FastAPI que carrega o modelo e atende as requisições
├── index.html                  # Interface gráfica para entrada dos dados e exibição do resultado
├── train_model.py              # Script de treinamento e exportação do modelo
├── personality_dataset.csv     # Base de dados original
├── personality_model.pkl       # Modelo treinado salvo
├── requirements.txt            # Dependências do projeto
└── README.md                   # Documentação do projeto
```

---

## Como Executar o Projeto

### 1. Acessar o Diretório do Projeto
```bash
cd Classificador-Comportamental
```

### 2. Configurar o Ambiente Virtual

**No Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**No macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as Dependências
```bash
pip install -r requirements.txt
```

### 4. Treinar o Modelo (Opcional)
Caso queira rodar o treinamento novamente e gerar o arquivo `personality_model.pkl`:
```bash
python train_model.py
```

### 5. Iniciar a API e a Aplicação
Inicie o servidor com o Uvicorn:
```bash
uvicorn app:app --reload
```
*(ou execute `python app.py`)*

### 6. Acessar no Navegador
* **Interface Gráfica:** Abra [http://localhost:8000](http://localhost:8000) no seu navegador.
* **Documentação Swagger:** Acesse [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Detalhes da API

### **Endpoint de Predição**
* **Rota:** `/api/v1/predict-personality`
* **Método:** `POST`
* **Payload de Entrada (JSON):**
  ```json
  {
    "time_spent_alone": 4.0,
    "stage_fear": 0,
    "social_event_attendance": 4.0,
    "going_outside": 3.0,
    "drained_after_socializing": 0,
    "friends_circle_size": 6.0,
    "post_frequency": 3.0
  }
  ```
* **Resposta de Sucesso (JSON):**
  ```json
  {
    "personality": "Extrovertido",
    "probability_extrovert": 71.0,
    "probability_introvert": 29.0,
    "confidence": 71.0
  }
  ```
