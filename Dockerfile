# 1. Start with a lightweight, modern base image containing GCC tools
FROM ubuntu:24.04

# 2. Install essential build tools and the HDF5 developer libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the active working directory inside the container image
WORKDIR /app

# 4. Copy your local project code into the container image
COPY . /app

# 5. Compile your C executable (linking the static/dynamic HDF5 dependencies)
# Using h5cc handles paths automatically, matching our previous thread
RUN (cd /app && make)

# 6. Make sure our Python entrypoint script has executable permissions
RUN chmod +x entrypoint.py

# 7. Tell Docker to route any run command arguments directly to the wrapper script
ENTRYPOINT ["python3", "entrypoint.py"]