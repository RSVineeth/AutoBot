FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install TA-Lib C library with more robust configuration
RUN cd /tmp && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr/local --build=x86_64-unknown-linux-gnu && \
    make -j$(nproc) && \
    make install && \
    cd /tmp && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib

# Update library cache and verify installation
RUN ldconfig -v && \
    ls -la /usr/local/lib/libta* && \
    ls -la /usr/local/include/ta-lib/

# Set environment variables for TA-Lib
ENV TA_LIBRARY_PATH=/usr/local/lib
ENV TA_INCLUDE_PATH=/usr/local/include
ENV LDFLAGS="-L/usr/local/lib"
ENV CPPFLAGS="-I/usr/local/include"
ENV PKG_CONFIG_PATH="/usr/local/lib/pkgconfig"

# Set working directory
WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Create a modified requirements.txt without TA-Lib for separate installation
RUN grep -v "TA-Lib" requirements.txt > requirements_without_talib.txt || cp requirements.txt requirements_without_talib.txt

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy Cython && \
    TALIB_INCLUDE_DIR=/usr/local/include TALIB_LIBRARY_DIR=/usr/local/lib pip install --no-cache-dir TA-Lib && \
    pip install --no-cache-dir -r requirements_without_talib.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]