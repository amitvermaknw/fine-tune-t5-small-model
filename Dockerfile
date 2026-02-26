FROM python:latest
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} 
ENTRYPOINT [ "python", "src/main.py" ]