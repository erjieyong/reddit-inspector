# Use the official lightweight Python image from
# https://hub.docker.com/_/python
FROM python:3.8-slim 

# Copy all the files needed for the app to work
COPY inference.py .
COPY models/ ./models/
COPY requirements.txt .

# Install all the necessary libraries
RUN pip install -r requirements.txt
RUN python -m textblob.download_corpora
RUN python -m spacy download en_core_web_sm

# Run the API!
CMD python inference.py
