FROM python:3.10.6-slim
COPY ~/.aws/credentials ~/.aws/credentials 
COPY app_utils.py /app
COPY requirements.txt /app
COPY app_utils.py /app
COPY main.py /app
COPY rm_model.pkl /app

# Install AWS CLI in DOCKER
RUN apt-get update && apt-get install -y curl
RUN apt-get install -y unzip
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install


# SET Credentials
ARG AWS_KEY= 'AKIAQA5VD6H3OWTF2UFU'
ARG AWS_SECRET_KEY= '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' 
ARG AWS_REGION= 'ap-south-1' 

RUN aws configure set aws_access_key_id $AWS_KEY \ 
&& aws configure set aws_secret_access_key $AWS_SECRET_KEY \
&& aws configure set default.region $AWS_REGION 

#Test case
COPY test.py /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN python test.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]