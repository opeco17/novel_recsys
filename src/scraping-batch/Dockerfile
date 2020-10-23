FROM python:3.8.0

RUN mkdir /var/www

WORKDIR /var/www

COPY requirements.txt ./

RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y vim && \
  pip install -r requirements.txt

COPY . /var/www

WORKDIR /var/www/src/main
