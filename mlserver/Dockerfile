FROM python:3.8.0

RUN mkdir /var/www
RUN mkdir /var/www/log

WORKDIR /var/www

COPY requirements.txt ./

RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y git && \
  apt-get install -y vim && \
  pip install -r requirements.txt

WORKDIR /var/www/src

CMD ["uwsgi","--ini","/var/www/uwsgi.ini"]
