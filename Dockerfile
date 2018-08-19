FROM pytorch:latest

RUN apt-get update && apt-get install -y screen

RUN if [ ! -d /workspace ]; then mkdir /workspace; fi
COPY ./requirements.txt /workspace/
RUN pip install -r /workspace/requirements.txt

ENV VISDOM_PORT 8098

CMD screen -dmS visdom python -m visdom.server -port $VISDOM_PORT ; /bin/bash