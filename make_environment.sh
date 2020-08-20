#!/bin/bash

docker network create narou_network

docker-compose up -d --build
