FROM python:3.12-slim

WORKDIR ./loss-app

RUN apt update && apt upgrade

COPY ./requirements.txt .
COPY ./*.py .

RUN pip3 install -r requirements.txt
