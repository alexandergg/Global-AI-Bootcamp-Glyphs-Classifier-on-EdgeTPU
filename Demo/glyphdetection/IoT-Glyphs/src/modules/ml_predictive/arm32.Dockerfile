FROM balenalib/raspberrypi3:stretch

RUN [ "cross-build-start" ]

RUN install_packages \
    python3 \
    python3-pip \
    python3-dev

COPY requirements.txt ./
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install -r requirements.txt

RUN install_packages \
    libboost-python1.62.0 \
    curl \
    libcurl4-openssl-dev

RUN install_packages \
    libatlas-base-dev \
    libopenjp2-7 \
    libtiff-tools \
    i2c-tools \
    libsm6 \
    libxext6 \
    libwebp-dev \
    libpng16-16 \
    libjasper-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libilmbase-dev \
    libopenexr-dev \
    libgstreamer1.0-dev \
    libqtgui4 \
    libqt4-test 

RUN rm -rf /var/lib/apt/lists/* \
    && apt-get -y autoremove

RUN [ "cross-build-end" ]  

ADD /app/ .

ENTRYPOINT [ "python3", "-u", "./main.py" ]