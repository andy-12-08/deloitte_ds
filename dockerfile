# Base image
FROM python:latest

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY . .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y cmake \
    build-essential \
    python3-distutils \
    && apt-get clean

# Install pyarrow from pip, which includes pre-built binaries
RUN pip install pyarrow

# Install remaining dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Set working directory for the app
WORKDIR /app/app

# Expose Streamlit port
EXPOSE 8503

# Set entrypoint and default command
ENTRYPOINT ["streamlit", "run", "--server.port", "8503"]
CMD ["main.py"]
