FROM continuumio/miniconda3

SHELL ["/bin/bash", "-c"]

WORKDIR /app

COPY ./ .

RUN apt-get update
RUN apt-get -y install make
RUN make setup

ENTRYPOINT ["/bin/bash", "-c"]
