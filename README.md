# Deploy ML models with FastAPI, Docker, and Heroku

## 1. Develop and save the model with this Colab
[Open Colab](https://colab.research.google.com)

## 2. Create Docker container

```bash
docker build -t app-name .
docker run -p 80:80 app-name
