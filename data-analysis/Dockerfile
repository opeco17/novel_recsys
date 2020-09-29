FROM jupyter/scipy-notebook

USER root

COPY requirements.txt ./

RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y git && \
  apt-get install -y vim && \
  apt-get install -y mecab && \
  apt-get install -y libmecab-dev && \
  apt-get install -y mecab-ipadic-utf8 && \
  pip install -r requirements.txt

COPY . ./