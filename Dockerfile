# Use a base image that already has TA-Lib compiled
FROM continuumio/miniconda3:latest

# Install Python 3.13 and necessary packages via conda
RUN conda install -c conda-forge python=3.13 libta-lib ta-lib numpy pandas -y && \
    conda clean -a

# Install pip packages
RUN pip install --no-cache-dir gunicorn

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Remove TA-Lib from requirements if it exists and install remaining packages
RUN grep -v "TA-Lib\|ta-lib\|talib" requirements.txt > requirements_filtered.txt || cp requirements.txt requirements_filtered.txt
RUN pip install --no-cache-dir -r requirements_filtered.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]