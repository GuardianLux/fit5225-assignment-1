FROM ubuntu:18.04
RUN apt-get update -y && apt-get install -y python3-pip python3-dev libsm6 libxext6 libxrender-dev
COPY ./requirements.txt /iWebLens_server/requirements.txt
WORKDIR /iWebLens_server
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY . /iWebLens_server
ENTRYPOINT [ "python3" ]
CMD [ "iWebLens_server.py" ]