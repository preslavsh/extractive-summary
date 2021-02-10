FROM python:3.9.0
COPY . /code
WORKDIR /code
RUN apt upgrade
RUN pip install --upgrade pip

# RUN pip install wheel -r requirements.txt