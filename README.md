### 🐳 Despliegue con Docker Compose

Este proyecto contiene dos servicios principales:

- `api`: Servicio backend para la inferencia del modelo.
- `web`: Interfaz web para interacción con el modelo desplegado.

#### ⚙️ Estructura del `docker-compose.yml`

```yaml
version: '3.8'

services:
  api:
    image: fraud-classification-tfm:latest
    build:
      context: ./api
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      API_PORT: 8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/docs"]
      interval: 5s
      timeout: 3s
      retries: 5

  web:
    image: web-fraud-classification-tfm
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    depends_on:
      api:
        condition: service_healthy
    environment:
      WEB_APP_PORT: 8080
      API_HOST: api
      API_PORT: 8000
▶️ Cómo ejecutarlo
Desde la raíz del repositorio, ejecuta:

bash
Copiar
Editar
docker-compose up --build
Luego, accede a:

API: http://localhost:8000/docs

Web: http://localhost:8080

Asegúrate de que ./api y ./frontend contengan sus respectivos Dockerfile y requirements.txt.
