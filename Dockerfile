FROM python:3.12

WORKDIR /app

COPY flask_app/ /app/

COPY models/lr_model.pkl /app/models/lr_model.pkl

RUN pip install -r requirements.txt

# RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

#local
CMD ["python", "app.py"]  

#Prod
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]