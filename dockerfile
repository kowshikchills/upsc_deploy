FROM python:3.10.6-slim
COPY ~/.aws/credentials ~/.aws/credentials 
COPY app_utils.py /app
COPY requirements.txt /app
COPY app_utils.py /app
COPY main.py /app
COPY rm_model.pkl /app

#RUN apt-get update && apt-get install -y curl
#RUN apt-get install -y unzip
#RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
#RUN unzip awscliv2.zip
#RUN ./aws/install
#RUN aws configure

COPY test.py /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN python test.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]