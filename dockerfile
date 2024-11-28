# Use a stable Python version
FROM python:3.10

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

# Upgrade pip and build tools to ensure compatibility
RUN pip install --upgrade pip setuptools wheel

# Install specific version of setuptools to fix the pkg_resources issue
RUN pip install setuptools>=65.0.0

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Set working directory for the app
WORKDIR /app/app

# Expose Streamlit port
EXPOSE 8503

# Set entrypoint and default command
ENTRYPOINT ["streamlit", "run", "--server.port", "8503"]
CMD ["main.py"]
