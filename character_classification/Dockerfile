FROM tensorflow/tensorflow:latest-gpu

MAINTAINER  Siddharth Yadav "siddharth16268@iiitd.ac.in"

# If needed, install system dependencies here
RUN apt-get update && apt-get install -y wget

# Starts installing python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt


WORKDIR /app


CMD ["bash"]