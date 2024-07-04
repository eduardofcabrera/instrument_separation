FROM python:3.10

ARG PYTHON_VERSION=3.10

RUN  python -m pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN python -m pip install  -r requirements.txt

COPY requirements-torch.txt requirements-torch.txt
RUN python -m pip install -r requirements-torch.txt

RUN apt-get update -y
RUN apt-get install -y libsox-dev

RUN rm requirements.txt
RUN rm requirements-torch.txt