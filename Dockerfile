FROM python:3.13-alpine

# Install build dependencies
RUN apk add --no-cache \
    build-base \
    wget \
    linux-headers \
    musl-dev \
    gcc \
    g++

# Download and compile TA-Lib
RUN cd /tmp && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -zxvf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --build=x86_64-alpine-linux-musl --prefix=/usr/local && \
    make && make install && \
    cd / && rm -rf /tmp/ta-lib*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy && \
    pip install --no-cache-dir TA-Lib && \
    grep -v "TA-Lib" requirements.txt > requirements_clean.txt && \
    pip install --no-cache-dir -r requirements_clean.txt

# Copy app
COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]