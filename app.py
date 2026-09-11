import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(
    title="Classificador Comportamental",
    description="Microsserviço de classificação de personalidade (Extrovertido vs. Introvertido)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "personality_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Erro ao carregar o modelo de '{MODEL_PATH}': {e}")


class PersonalityInputData(BaseModel):
    time_spent_alone: float = Field(..., ge=0, le=24, example=4.0, description="Horas diárias em solitude")
    stage_fear: int = Field(..., ge=0, le=1, example=0, description="1 para medo de palco/público, 0 caso contrário")
    social_event_attendance: float = Field(..., ge=0, le=10, example=4.0, description="Frequência em eventos sociais (escala 0 a 10)")
    going_outside: float = Field(..., ge=0, le=7, example=3.0, description="Dias por semana que sai de casa (0 a 7)")
    drained_after_socializing: int = Field(..., ge=0, le=1, example=0, description="1 se sente drenado após socializar, 0 caso contrário")
    friends_circle_size: float = Field(..., ge=0, le=50, example=6.0, description="Tamanho do círculo de amigos próximos")
    post_frequency: float = Field(..., ge=0, le=10, example=3.0, description="Frequência de postagens em redes sociais (escala 0 a 10)")


class PersonalityPredictionResponse(BaseModel):
    personality: str = Field(..., description="Classificação binária: 'Extrovertido' ou 'Introvertido'")


@app.get("/", response_class=FileResponse)
def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Arquivo index.html não encontrado no diretório do projeto.")
    return FileResponse(index_path)


@app.post("/api/v1/predict-personality", response_model=PersonalityPredictionResponse)
def predict_personality(data: PersonalityInputData):
    features = pd.DataFrame([{
        'Time_spent_Alone': data.time_spent_alone,
        'Stage_fear': data.stage_fear,
        'Social_event_attendance': data.social_event_attendance,
        'Going_outside': data.going_outside,
        'Drained_after_socializing': data.drained_after_socializing,
        'Friends_circle_size': data.friends_circle_size,
        'Post_frequency': data.post_frequency
    }])

    try:
        prediction = int(model.predict(features)[0])
        personality = "Extrovertido" if prediction == 1 else "Introvertido"

        return PersonalityPredictionResponse(personality=personality)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na inferência do modelo: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
