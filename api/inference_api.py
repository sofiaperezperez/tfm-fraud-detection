#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference_api.py: API para servir predicciones del modelo de precios de casas utilizando FastAPI.
Carga los artefactos (modelo, encoders y scaler) y define un endpoint para recibir datos en JSON
y retornar la predicción.
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import torch.nn as nn
import numpy as np 
import os

from openai import OpenAI
from dotenv import dotenv_values

API_PORT = int(os.environ.get("API_PORT", 8000))


def generar_prompt_llm(prediccion, shap_local, shap_global, valores_registro, valores_registro_todas_las_vars,dict_definiciones, top_n=5):
    top_local = sorted(shap_local.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    top_global = sorted(shap_global.items(), key=lambda x: x[1], reverse=True)[:3]

    prompt = (
        f"Un modelo de red neuronal ha predicho una probabilidad de fraude del {prediccion:.2f} en el momento de apertura de una cuenta bancaria. Cuanto mas cercano a 0, menos probable, y cuanto mas cercano a 1 es mas probable a ser fraude.\n\n"
        f"Las {top_n} variables que más influyen en esta predicción son:\n"
    )
    for feat, val_shap in top_local:
        if "_enc" in feat:
            valor_real = valores_registro_todas_las_vars[feat[:-4]]
        else:
            valor_real = valores_registro.get(feat, "N/A")
        signo = "+" if val_shap >= 0 else ""
        prompt += f"- `{feat} = {valor_real}`, con contribución de {signo}{val_shap:.2f}. \n"

    prompt += "Ten en cuenta que si la contribucion es mayor que 0 la contribucion es positiva y si no negativa." 
    prompt += "Ten en cuenta que si la contribución es negativa, contribuye a que el registro no sea fraude."
    prompt += "\nDe forma global, las variables más importantes para el modelo son:\n"
    for i, (feat, imp) in enumerate(top_global, 1):
        prompt += f"{i}. `{feat}` (importancia media = {imp:.2f})\n"
        prompt += f"La definicion de la variable `{feat}` es:  {dict_definiciones[feat]}\n"

    prompt += (
        "\nPor favor, imagina que trabajas en un banco y explica, de manera completa y original, sin repetir datos, por qué el modelo considera que esta apertura de cuenta tiene una alta o baja probabilidad de ser fraude.Usa ejemplos, metáforas o situaciones reales para ilustrar. "
        "Describe el posible impacto de cada variable y qué significa cada contribución positiva en el contexto de detección de fraude. "
        "Usa lenguaje claro y ejemplos sencillos para que alguien sin conocimientos técnicos pueda entender. Asegurate de terminar las frases y se conciso."
    )
    return prompt


class TabNetWrapper(nn.Module):
    def __init__(self, tabnet_network):
        super().__init__()
        self.network = tabnet_network

    def forward(self, x):
        # x: tensor float32 (batch, features)
        out, _ = self.network(x)
        return out  # logits, tensor con grad_fn
    
# Definir el esquema de datos para la petición de inferencia
class FraudData(BaseModel):
    income : float
    name_email_similarity : float
    prev_address_months_count : float
    current_address_months_count : float
    customer_age : float
    days_since_request : float
    intended_balcon_amount : float
    zip_count_4w : float
    velocity_24h : float
    velocity_4w : float
    date_of_birth_distinct_emails_4w : float
    credit_risk_score : float
    bank_months_count : float
    proposed_credit_limit : float
    session_length_in_minutes : float
    device_distinct_emails_8w : float
    payment_type : str
    employment_status : str
    housing_status : str
    source : str
    device_os : str
    keep_alive_session : int
    email_is_free : int
    phone_home_valid : int
    phone_mobile_valid : int
    has_other_cards : int
    foreign_request : int

# Cargar artefactos guardados
clf = TabNetClassifier()
clf.load_model("api/artifacts/modelo_tabnet_reduccion_vars.zip")
model_wrapper = torch.load("api/artifacts/model_wrapper_full.pth", weights_only=False)
model_wrapper.eval()
explainer = joblib.load('api/artifacts/explainer_gradient_shap_tabnet_reduccion_vars.pkl')
encoder = joblib.load("api/artifacts/target_encoder.pkl")
# Columnas categóricas y numéricas (según el preprocesamiento)
cols_categ = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os', 
              'keep_alive_session', 'email_is_free', 
              'phone_home_valid', 'phone_mobile_valid', 'has_other_cards', 'foreign_request']

columnas_train_reducidas = ['income', 'name_email_similarity', 'prev_address_months_count', 
                            'current_address_months_count', 'customer_age', 'days_since_request', 
                            'intended_balcon_amount', 'zip_count_4w', 'velocity_24h', 'velocity_4w',
                            'date_of_birth_distinct_emails_4w', 'credit_risk_score', 
                            'bank_months_count', 'proposed_credit_limit', 'session_length_in_minutes', 
                            'device_distinct_emails_8w', 'payment_type_enc', 'employment_status_enc', 
                            'housing_status_enc', 'device_os_enc', 'keep_alive_session_enc', 'email_is_free_enc',
                            'phone_home_valid_enc', 'phone_mobile_valid_enc', 'has_other_cards_enc', 
                            'foreign_request_enc']

# Inicializar la aplicación FastAPI
app = FastAPI(title="API de Inferencia - Predicción de fraude")

@app.post("/predict_fraud")
async def predict_fraud(data: FraudData):
    """
    Endpoint para recibir datos de una casa y retornar la predicción del precio.
    """
    # Convertir el objeto recibido a DataFrame con model_dump
    input_data = pd.DataFrame([data.model_dump()])
    df_encoded = encoder.transform(input_data[cols_categ])
    # Asignamos las columnas codificadas a un nuevo DataFrame con el sufijo '_enc'
    for col in cols_categ:
        input_data[col + '_enc'] = df_encoded[col]
    
    # Realizar la predicción
    fila_tensor = torch.tensor(np.array(input_data[columnas_train_reducidas]), dtype=torch.float32)
    preds_proba_instance = clf.predict_proba(fila_tensor)[:, 1]

    shap_values_registro = explainer.shap_values(fila_tensor)
    shap_vals_clase1_registro = shap_values_registro[0, :, 1] 

    # 1. Crear shap_local (valores SHAP por feature en el registro)
    shap_registro = {feature: float(val) for feature, val in zip(columnas_train_reducidas, shap_vals_clase1_registro)}

    # 2. Crear valores_registro (valores reales de la instancia)
    features_np = fila_tensor.cpu().numpy()[0] if hasattr(fila_tensor, "cpu") else fila_tensor.numpy()[0]

    valores_registro = {feature: float(val) for feature, val in zip(columnas_train_reducidas, features_np)}
    valores_registro_todas_las_vars = input_data.to_dict(orient='records')[0]
    #shap_global_registro = {feature: float(val) for feature, val in zip(columnas_train_reducidas, importances_clase1)}
    shap_global_registro = {'income': 0.08911339063836488,
                            'name_email_similarity': 0.11564854495112464,
                            'prev_address_months_count': 0.1279070045363714,
                            'current_address_months_count': 0.042791404254931206,
                            'customer_age': 0.05370417907543024,
                            'days_since_request': 0.01169516899954716,
                            'intended_balcon_amount': 0.015278193141903273,
                            'zip_count_4w': 0.037472867342504476,
                            'velocity_24h': 0.0083370016784055,
                            'velocity_4w': 0.10728575851091357,
                            'date_of_birth_distinct_emails_4w': 0.04721438820890298,
                            'credit_risk_score': 0.06136171234820919,
                            'bank_months_count': 0.0932074424736741,
                            'proposed_credit_limit': 0.025648505211855202,
                            'session_length_in_minutes': 0.024425766262507646,
                            'device_distinct_emails_8w': 0.03991915449219584,
                            'payment_type_enc': 0.09706930179957525,
                            'employment_status_enc': 0.0798355924385771,
                            'housing_status_enc': 0.26107936379313407,
                            'device_os_enc': 0.19518044345850952,
                            'keep_alive_session_enc': 0.12698692821207294,
                            'email_is_free_enc': 0.07991560498635195,
                            'phone_home_valid_enc': 0.1902635249485717,
                            'phone_mobile_valid_enc': 0.02070590280904,
                            'has_other_cards_enc': 0.13482324751250502,
                            'foreign_request_enc': 0.009745362898078693}



    dict_definiciones = {'income': "Decil en el que se encuentra el ingreso anual del cliente",
    'name_email_similarity': "Similitud entre el email y nombre del aplicante",
    'prev_address_months_count': "Número de meses registrados en la casa anterior del cliente ",
    'current_address_months_count': "Meses registrados en la casa actual",
    'customer_age': "Edad en años del cliente, redondeada a la década ",
    'days_since_request': "Número de días desde que se hizo la solicitud de la apertura",
    'intended_balcon_amount':"Cantidad inicial transferida para la solicitud ",
    'zip_count_4w': "Cuantas solicitudes se han hecho en el mismo código postal en las últimas 4 semanas. Si es muy alta significa que ha hecho muchos intentos de solicitudes.",
    'velocity_24h': "Cuantas solicitudes ha hecho el cliente por hora en las últimas 24 horas. Si es muy alta significa que ha hecho muchos intentos de solicitudes.",
    'velocity_4w': "Cuantas solicitudes ha hecho el cliente cada hora en las últimas 4 semanas. Si es muy alta significa que ha hecho muchos intentos de solicitudes.",
    'date_of_birth_distinct_emails_4w': "Cuantos emails hay registrados con la misma fecha de nacimiento. Si es muy alta significa que hay muchos intentos de solicitudes.",
    'credit_risk_score': "Puntuación del cliente por calidad crediticia",
    'bank_months_count': "Antigüedad en meses de la cuenta bancaria",
    'proposed_credit_limit': "Límite de crédito solicitado",
    'session_length_in_minutes': "Cuanto tiempo estuvo en la web haciendo la solicitud el cliente",
    'device_distinct_emails_8w': "Cuantos emails distintos se han registrado desde el mismo dispositivo",
    'payment_type_enc': "Plan de pago con tarjeta de crédito",
    'employment_status_enc': "Indicador de empleo del cliente",
    'housing_status_enc': "Estado residencial",
    'device_os_enc': "Sistema operativo usado en la aplicacion",
    'keep_alive_session_enc':"Si el cliente quiere mantener la sesion activa o no",
    'email_is_free_enc': "Dominio del email",
    'phone_home_valid_enc': "Si el teléfono fijo usado es válido",
    'phone_mobile_valid_enc': "Si el teléfono movil usado es válido",
    'has_other_cards_enc': "Si tiene otras tarjetas del mismo banco",
    'foreign_request_enc': "Si la solicitud proviene de otro país"}

    prompt = generar_prompt_llm(preds_proba_instance[0],shap_registro, shap_global_registro, valores_registro, 
                                valores_registro_todas_las_vars, dict_definiciones)

    print(prompt)

    ENV_CONFIG = dotenv_values(".env")
    API_KEY = ENV_CONFIG.get("OPENAI_API_KEY")
    SERVER_URL = ENV_CONFIG.get("SERVER_URL", "http://http://g4.etsisi.upm.es:8833/v1")
    MODEL_ID = ENV_CONFIG.get("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

    client = OpenAI(
        api_key=API_KEY,
        base_url=SERVER_URL,
    )

    USER_PROMPT_TEMPLATE = prompt.strip()

    messages = [
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(name="Pablo")
        }
    ]
          
    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=500,
        temperature=0.9,
    )

    print("Response:")
    for choice in completion.choices:
        print(choice.message.content.strip())


    # Retornar la predicción  float(preds_proba_instance[0]),
    return {"predicted_fraud":  choice.message.content.strip(), 
            "shap_registro":shap_registro, 
            "valores_registro": valores_registro}


# Ejecutar la API (ejecutar este script directamente)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
