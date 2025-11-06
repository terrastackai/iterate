FROM quay.io/fedora/fedora:44

# Install necessary build tools and dependencies for Python
# - build-essential (or equivalent on Fedora) provides essential build tools
# - libffi-devel, openssl-devel, bzip2-devel, libsqlite3x-devel, etc. are common Python build dependencies
RUN dnf update -y && \
    dnf install -y \
        gcc \
        clang \
        clang-tools-extra \
        make \
        wget \
        tar \
        libffi-devel \
        openssl-devel \
        bzip2-devel \
        libsqlite3x-devel \
        zlib-devel \
        readline-devel \
        xz-devel \
        gdbm-devel \
        tk-devel \
        libuuid-devel \
        ncurses-devel \
        python3-pip \
        python3-devel && \
    dnf clean all

# Set a working directory
WORKDIR /usr/src/python

# Download Python 3.13 source code
ENV PYTHON_VERSION 3.13.1
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    rm Python-${PYTHON_VERSION}.tgz

# Build and install Python 3.13
WORKDIR /usr/src/python/Python-${PYTHON_VERSION}
RUN ./configure --enable-optimizations --prefix=/usr/local && \
    make -j $(nproc) && \
    make install

# Set Python 3.13 as the default Python interpreter
ENV PATH="/usr/local/bin:$PATH"

# Verify installation
RUN python3.13 --version
RUN pip3.13 --version
RUN pip3.13 install --upgrade pip
# You can add further commands here to install your application dependencies

CMD ["python3.13"]
WORKDIR /app
COPY . /app/
# RUN pip3.13 install scipy -U
RUN pip3.13 install -e .

# RUN python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ terratorch-iterate==0.2.2rc3

