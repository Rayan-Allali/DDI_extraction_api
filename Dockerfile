FROM python:3.8.10.slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
RUN python3 -m spacy download en_core_web_sm
CMD uvicorn main:app --port=8000 --host=0.0.0.0