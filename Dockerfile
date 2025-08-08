FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install TA-Lib C library
RUN cd /tmp && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    ldconfig && \
    cd /tmp && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib

# Set environment variables for TA-Lib
ENV TA_LIBRARY_PATH=/usr/local/lib
ENV TA_INCLUDE_PATH=/usr/local/include
ENV LDFLAGS="-L/usr/local/lib"
ENV CPPFLAGS="-I/usr/local/include"

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

# Create logs directory for the trading bot
RUN mkdir -p /app/logs

# Expose port
EXPOSE 5000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Run the application with proper gunicorn settings for your bot
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "--keep-alive", "2", "app:app"]