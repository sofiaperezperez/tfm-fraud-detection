import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
import torch 
import numpy as np
import torch.nn as nn

class TabNetWrapper(nn.Module):
    def __init__(self, tabnet_network):
        super().__init__()
        self.network = tabnet_network

    def forward(self, x):
        # x: tensor float32 (batch, features)
        out, _ = self.network(x)
        return out  # logits, tensor con grad_fn


import openai
import os

# Asegúrate de tener la variable de entorno configurada

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
        "Usa lenguaje claro y ejemplos sencillos para que alguien sin conocimientos técnicos pueda entender. Asegurate de terminar las frases y haz un pequeño resumen al final de las variables que mas peso positivo tienen."
    )
    return prompt


data = pd.read_csv('Base.csv', sep=',')

clf = TabNetClassifier()
registro = 11
clf.load_model("modelo_tabnet_reduccion_vars.zip")
model_wrapper = torch.load("model_wrapper_full.pth", weights_only=False)
model_wrapper.eval()
import joblib
explainer = joblib.load('explainer_gradient_shap_tabnet_reduccion_vars.pkl')
encoder = joblib.load('target_encoder.pkl')
cols_categ = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os', 
              'keep_alive_session', 'email_is_free', 
              'phone_home_valid', 'phone_mobile_valid', 'has_other_cards', 'foreign_request']

fila_np = data.iloc[[registro]]  # O X_test[[i]]

# Ajustamos y transformamos el DataFrame
df_encoded = encoder.transform(fila_np[cols_categ], fila_np['fraud_bool'])
# Asignamos las columnas codificadas a un nuevo DataFrame con el sufijo '_enc'
for col in cols_categ:
    fila_np[col + '_enc'] = df_encoded[col]

columnas_train_reducidas = ['income', 'name_email_similarity', 'prev_address_months_count', 
                            'current_address_months_count', 'customer_age', 'days_since_request', 
                            'intended_balcon_amount', 'zip_count_4w', 'velocity_24h', 'velocity_4w',
                              'date_of_birth_distinct_emails_4w', 'credit_risk_score', 
                              'bank_months_count', 'proposed_credit_limit', 'session_length_in_minutes', 
                              'device_distinct_emails_8w', 'payment_type_enc', 'employment_status_enc', 
                              'housing_status_enc', 'device_os_enc', 'keep_alive_session_enc', 'email_is_free_enc',
                                'phone_home_valid_enc', 'phone_mobile_valid_enc', 'has_other_cards_enc', 
                                'foreign_request_enc']
fila_tensor = torch.tensor(np.array(fila_np[columnas_train_reducidas]), dtype=torch.float32)


preds_proba_instance = clf.predict_proba(fila_tensor)[:, 1]

# Predicciones de clases (0 o 1)
preds_class_instance = clf.predict(fila_tensor)
print(f"La propensión es {preds_proba_instance[0]}")

shap_values_registro = explainer.shap_values(fila_tensor)
shap_vals_clase1_registro = shap_values_registro[0, :, 1] 

# 1. Crear shap_local (valores SHAP por feature en el registro)
shap_registro = {feature: float(val) for feature, val in zip(columnas_train_reducidas, shap_vals_clase1_registro)}

# 2. Crear valores_registro (valores reales de la instancia)
features_np = fila_tensor.cpu().numpy()[0] if hasattr(fila_tensor, "cpu") else registro[0]

valores_registro = {feature: float(val) for feature, val in zip(columnas_train_reducidas, features_np)}
valores_registro_todas_las_vars = data.iloc[registro] .to_dict()
#shap_global_registro = {feature: float(val) for feature, val in zip(columnas_train_reducidas, importances_clase1)}
shap_global_registro = {'income': 0.10793337526122135, 
                        'name_email_similarity': 0.14378745230508094, 
                        'prev_address_months_count': 0.06565491997919586, 
                        'current_address_months_count': 0.11455215395113443, 
                        'customer_age': 0.051517691039927864, 
                        'days_since_request': 0.013576690110434988, 
                        'intended_balcon_amount': 0.015758084586188655, 
                        'zip_count_4w': 0.03386604401398323, 
                        'velocity_24h': 0.012766773885222286, 
                        'velocity_4w': 0.044224360070094, 
                        'date_of_birth_distinct_emails_4w': 0.060755362857931296, 
                        'credit_risk_score': 0.06917560834020675, 
                        'bank_months_count': 0.037712042349504524, 
                        'proposed_credit_limit': 0.04417333600735399, 
                        'session_length_in_minutes': 0.02452420117030188, 
                        'device_distinct_emails_8w': 0.05309947108619462, 
                        'payment_type_enc': 0.0732105706574691, 
                        'employment_status_enc': 0.0661931346035797, 
                        'housing_status_enc': 0.23125106075763857, 
                        'device_os_enc': 0.14675092068342868, 
                        'keep_alive_session_enc': 0.1322881175114797, 
                        'email_is_free_enc': 0.0922074435447285, 
                        'phone_home_valid_enc': 0.18494930676543253, 
                        'phone_mobile_valid_enc': 0.023292036288802064, 
                        'has_other_cards_enc': 0.07290114070891279, 
                        'foreign_request_enc': 0.01832983407539991}



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


import os

from openai import OpenAI
from dotenv import dotenv_values

ENV_CONFIG = dotenv_values(".env")
API_KEY = ENV_CONFIG.get("OPENAI_API_KEY")
SERVER_URL = ENV_CONFIG.get("SERVER_URL", "http://http://g4.etsisi.upm.es:8833/v1")
MODEL_ID = ENV_CONFIG.get("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

client = OpenAI(
    api_key=API_KEY,
    base_url=SERVER_URL,
)

USER_PROMPT_TEMPLATE = prompt.strip()

def main():
    messages = [
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(name="Pablo")
        }
    ]
            
    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=800,
        temperature=0.9,
    )

    print("Response:")
    for choice in completion.choices:
        print(choice.message.content.strip())
            

if __name__ == "__main__":
    main()
