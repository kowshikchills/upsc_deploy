#docker build
sudo docker build -t <name> .
sudo docker run -p 80:80 upscdeploy

#push to ecr
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin XXXXXXXXX.dkr.ecr.ap-south-1.amazonaws.com
docker tag upscdeploy:latest XXXXXXXXXX.dkr.ecr.ap-south-1.amazonaws.com/alphamentordeploy:latest
docker push XXXXXXXXXXXXXXXX.dkr.ecr.ap-south-1.amazonaws.com/alphamentordeploy:latest


#ecr to ecs

