FROM python:3.10

WORKDIR /code

EXPOSE 8080

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV POETRY_VIRTUALENVS_CREATE=false

RUN pip3 install poetry

RUN apt-get update && apt-get install -y swig

COPY ./poetry.lock ./poetry.lock
COPY pyproject.toml ./pyproject.toml
COPY poetry.toml ./poetry.toml

RUN poetry install --no-interaction