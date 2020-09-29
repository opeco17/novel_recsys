FROM python:3.8.0

RUN mkdir /var/www

WORKDIR /var/www

COPY requirements.txt ./

RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y git && \
  apt-get install -y vim && \
  apt-get install -y mecab && \
  apt-get install -y libmecab-dev && \
  apt-get install -y mecab-ipadic-utf8 && \
  pip install -r requirements.txt

COPY . /var/www

WORKDIR /var/www/src/main

CMD ["uwsgi","--ini","/var/www/uwsgi.ini"]
