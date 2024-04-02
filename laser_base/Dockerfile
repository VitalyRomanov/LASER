FROM continuumio/miniconda3

MAINTAINER Gilles Bodart <gillesbodart@users.noreply.github.com>

WORKDIR /app
RUN git clone https://github.com/facebookresearch/LASER.git
ENV LASER /app/LASER
WORKDIR $LASER

# Install dependencies
RUN conda create -n env python=3.7
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

RUN apt-get -qq -y update
RUN apt-get -qq -y upgrade
RUN apt-get -qq -y install \
        gcc \
        g++ \
        wget \
        curl \
        git \
        make \
        unzip \
        sudo \
        vim \
        cmake

# Use C.UTF-8 locale to avoid issues with ASCII encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r /app/requirements.txt --verbose

#Installing FAISS

RUN conda install --name env -c pytorch faiss-cpu -y

RUN bash ./install_external_tools.sh

CMD ["/bin/bash"]
