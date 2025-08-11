# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 poppler-utils

# Copy requirements first for caching
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# --- Pre-cache models ---
ENV HF_HOME=/code/hf_cache
COPY ./download_models.py /code/download_models.py
RUN python download_models.py
# --- End pre-caching ---

# Now, copy the rest of your application code
COPY ./main.py /code/main.py

# Set environment variables for Hugging Face Spaces
ENV PORT=7860
ENV HOST=0.0.0.0

# Expose the port and run the application
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]