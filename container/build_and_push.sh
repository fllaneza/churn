#!/usr/bin/env bash

# Este script muestra como construir la imagen Docker y subirla a ECR para ser usada 
# por Amazon SageMaker

# El argumento de este script es el nombre de la imagen Docker que se construirá en local.
# Concatenandole la cuenta de AWS y la region se forma el nombre del repositorio de ECR
image=$1

if [[ "$image" == "" ]]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

# Convierte los scripts en ejecutables
chmod +x src/train
chmod +x src/serve

# Obtiene el número de cuenta de AWS asociado con los credenciales IAM actuales
account=$(aws sts get-caller-identity --query Account --output text)

if [[ $? -ne 0 ]]
then
    exit 255
fi

# Obtiene la región definida en la configuración actual (si no, ponemos eu-west-1)
region=$(aws configure get region)
region=${region:-eu-west-1}

# Nombre completo del repositorio de ECR
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# Comprueba si existe un repositorio con la imagen Docker, y si no se crea
aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [[ $? -ne 0 ]]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Hacemos login en ECR
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${fullname}

# Construimos la imagen Docker localmente con el mismo nombre y la subimos
# a ECR taggeada con el nombre completo del repositorio de ECR

docker build  -t ${image} .
docker tag ${image} ${fullname}
docker push ${fullname}
