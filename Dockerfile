# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# Install system dependencies required by libraries
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 poppler-utils

# --- OPTIMIZATION START ---
# First, copy only the requirements file
COPY ./requirements.txt /code/requirements.txt

# Install the Python packages. This layer will now be cached.
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
# --- OPTIMIZATION END ---

# Now, copy the rest of your application code
COPY . /code/

# Set environment variables for the app to run
ENV PORT=8080
ENV HOST=0.0.0.0

# Expose the port and run the application
EXPOSE $PORT
CMD uvicorn main:app --host 0.0.0.0 --port $PORT