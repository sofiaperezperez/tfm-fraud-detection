from flask import Flask, render_template, request
import requests
import os
import markdown 
import json
from openai import OpenAI
from dotenv import dotenv_values
app = Flask(__name__)

# Configuración de la URL base de la API
# Se puede configurar a través de variables de entorno (Docker) o por defecto localhost:8000 (se ejecuta sin Docker)
API_HOST = os.environ.get("API_HOST", "localhost")
API_PORT = os.environ.get("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}/predict_fraud"

# Puerto de la aplicación web
WEB_APP_PORT = os.environ.get("WEB_APP_PORT", "8080")


def construir_prompt_modificaciones(local_shap: dict, valores_actuales: dict, umbral_shap=0) -> str:
    # Filtrar variables con impacto SHAP positivo significativo
    variables_importantes = {k: v for k, v in local_shap.items() if v > umbral_shap}

    # Ordenar por impacto descendente
    variables_ordenadas = dict(sorted(variables_importantes.items(), key=lambda item: item[1], reverse=True))

    rangos_variables = {'income': "Rango permitido: de 0 a 1. Es menos fraudulento si está en torno a 0.5",
    'name_email_similarity': "Toma valores entre 0 y 1. Es menos fraudulento si es un valor menor.",
    'prev_address_months_count': "Toma valores de 0 a infinito. Es menos fraudulento si ",
    'current_address_months_count': "Toma valores entre 0 y infinito. Es menos fraudulento si es mayor.",
    'customer_age': "Rango permitido de 10 a 90. Es menos fraudulento si es un valor mayor.",
    'days_since_request': "Rango de 0 a infinito.",
    'intended_balcon_amount':"Rango de -16 a infinito. ",
    'zip_count_4w': "de 1 a infinito. Es más fraudulento cuanto mayor valor.",
    'velocity_24h': "de 1 a infinito.",
    'velocity_4w': "de 1 a infinito. Es más fraudulento cuanto mayor valor.",
    'date_of_birth_distinct_emails_4w': "de 1 a infinito. Es más fraudulento cuanto menor valor.",
    'credit_risk_score': "de -191 a 389. Es más fraudulento cuanto mayor valor.",
    'bank_months_count': "De 1 a infinito. Es más fraudulento cuanto mayor valor.",
    'proposed_credit_limit': "De 0 a infinito. Es más fraudulento cuanto mayor valor.",
    'session_length_in_minutes': "De 0 a infinito",
    'device_distinct_emails_8w': "De 1 a infinito. Es más fraudulento cuanto mayor valor.",
    'payment_type_enc': "Plan de pago con tarjeta de crédito",
    'employment_status_enc': "Indicador de empleo del cliente. El menos fraudulento es AE.",
    'housing_status_enc': "Estado residencial. El menos fraudulento es BE.",
    'device_os_enc': "Sistema operativo usado en la aplicacion. El menos fraudulento es linux.",
    'keep_alive_session_enc':"Si el cliente quiere mantener la sesion activa o no. Es menos fraudulento un 1 que un 0.",
    'email_is_free_enc': "Dominio del email. Es mas fraudulento un 1 que un 0",
    'phone_home_valid_enc': "Si el teléfono fijo usado es válido. Es menos fraudulento un 1 que un 0",
    'phone_mobile_valid_enc': "Si el teléfono movil usado es válido. Es menos fraudulento un 1 que un 0",
    'has_other_cards_enc': "Si tiene otras tarjetas del mismo banco. Es menos fraudulento un 1 que un 0",
    'foreign_request_enc': "Si la solicitud proviene de otro país. Es mas fraudulento un 1 que un 0"}



    prompt = "Analiza las variables que más contribuyen positivamente a la propensión al fraude:\n\n"

    prompt += "Variables con impacto SHAP positivo:\n"
    for feature, shap_value in variables_ordenadas.items():
        prompt += f"- {feature}: impacto SHAP = {shap_value}\n"

    prompt += "\nValores actuales de estas variables:\n"
    for feature in variables_ordenadas.keys():
        valor = valores_actuales.get(feature, "No disponible")
        prompt += f"- {feature}: valor actual = {valor}\n"

    prompt += "\nRangos recomendados o aceptables para estas variables:\n"
    for feature in variables_ordenadas.keys():
        rango = rangos_variables.get(feature, "No disponible")
        if isinstance(rango, (tuple, list)) and len(rango) == 2:
            prompt += f"- {feature}: rango permitido entre {rango[0]} y {rango[1]}\n"
        else:
            prompt += f"- {feature}: {rango}\n"

    prompt += (
        "\nBasándote en esta información, indica qué variables y qué valores deberían tomar para reducir la probabilidad de fraude. "
        "Se conciso."
    )
    print(prompt)
    return prompt



@app.route("/", methods=["GET"])
def index():
    # Renderiza un formulario HTML
    return render_template("index.html")

@app.route("/predict_fraud", methods=["POST"])
def predict_fraud():
    # Recogemos los datos del formulario
    income = request.form.get("income")            
    name_email_similarity = request.form.get("name_email_similarity") 
    prev_address_months_count = request.form.get("prev_address_months_count") 
    current_address_months_count = request.form.get("current_address_months_count") 
    customer_age = request.form.get("customer_age") 
    days_since_request = request.form.get("days_since_request") 
    intended_balcon_amount = request.form.get("intended_balcon_amount") 
    zip_count_4w = request.form.get("zip_count_4w") 
    velocity_24h = request.form.get("velocity_24h") 
    velocity_4w = request.form.get("velocity_4w")
    date_of_birth_distinct_emails_4w = request.form.get("date_of_birth_distinct_emails_4w") 
    credit_risk_score = request.form.get("credit_risk_score") 
    bank_months_count = request.form.get("bank_months_count") 
    proposed_credit_limit = request.form.get("proposed_credit_limit") 
    session_length_in_minutes = request.form.get("session_length_in_minutes") 
    device_distinct_emails_8w = request.form.get("device_distinct_emails_8w") 
    payment_type = request.form.get("payment_type") 
    employment_status = request.form.get("employment_status") 
    housing_status = request.form.get("housing_status") 
    source = request.form.get("source") 
    device_os = request.form.get("device_os") 
    keep_alive_session = request.form.get("keep_alive_session") 
    email_is_free = request.form.get("email_is_free") 
    phone_home_valid = request.form.get("phone_home_valid") 
    phone_mobile_valid = request.form.get("phone_mobile_valid") 
    has_other_cards = request.form.get("has_other_cards") 
    foreign_request = request.form.get("foreign_request")

    # Construimos la carga en JSON para la API
    payload = {
        "income" : float(income),
        "name_email_similarity" : float(name_email_similarity),
        "prev_address_months_count" : float(prev_address_months_count),
        "current_address_months_count" : float(current_address_months_count),
        "customer_age" : float(customer_age),
        "days_since_request" : float(days_since_request),
        "intended_balcon_amount" : float(intended_balcon_amount),
        "zip_count_4w" : float(zip_count_4w),
        "velocity_24h" : float(velocity_24h),
        "velocity_4w" : float(velocity_4w),
        "date_of_birth_distinct_emails_4w" : float(date_of_birth_distinct_emails_4w),
        "credit_risk_score" : float(credit_risk_score),
        "bank_months_count" : float(bank_months_count),
        "proposed_credit_limit" : float(proposed_credit_limit),
        "session_length_in_minutes" : float(session_length_in_minutes),
        "device_distinct_emails_8w" : float(device_distinct_emails_8w),
        "payment_type" : str(payment_type),
        "employment_status" : str(employment_status),
        "housing_status" : str(housing_status),
        "source" : str(source),
        "device_os" : str(device_os),
        "keep_alive_session" : int(keep_alive_session),
        "email_is_free" : int(email_is_free),
        "phone_home_valid" : int(phone_home_valid),
        "phone_mobile_valid" : int(phone_mobile_valid),
        "has_other_cards" : int(has_other_cards),
        "foreign_request" : int(foreign_request)}
    

    # Hacemos la petición POST a la API
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        predicted_price = data["predicted_fraud"]
        html_text = markdown.markdown(predicted_price)

        return render_template("resultado.html", prediccion=html_text, 
                               shap_values = data['shap_registro'], 
                               valores_registro = data['valores_registro'])
        #return f"Información sobre la solicitud realizada: {predicted_price}"
    else:
        return "Error en la API. No se pudo obtener la predicción."


@app.route('/posibles_cambios', methods=['POST'])
def suggest_changes():
    # Recogemos el JSON de SHAP
    shap_json = request.form['local_shap_json']
    local_shap = json.loads(shap_json)  # dict {feature: valor}

    valores_vars_json = request.form['valores_registro_json']
    valores_vars = json.loads(valores_vars_json) 

    # Construye aquí el prompt para el LLM usando local_shap
    prompt = construir_prompt_modificaciones(local_shap, valores_vars)
    
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
        max_tokens=400,
        temperature=0.9,
    )

    print("Response:")
    for choice in completion.choices:
        print(choice.message.content.strip())
    sugerencias_html_text = markdown.markdown(choice.message.content.strip())


    # Renderiza la plantilla de sugerencias
    return render_template('sugerencias.html', sugerencias=sugerencias_html_text)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=WEB_APP_PORT, debug=True)
