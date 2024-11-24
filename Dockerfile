FROM python:3.11

WORKDIR /app

ENV PYTHONUNBUFFERED True

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

ENV PORT=8080  

EXPOSE ${PORT}

COPY . ./app
    
ENTRYPOINT streamlit run --server.port ${PORT} app/app.py –-server.address=0.0.0.0

#docker build -t streamlit:latest -f Dockerfile .
#docker run --name streamlit-container --network my_network -p 8080:8080 streamlit:latest
