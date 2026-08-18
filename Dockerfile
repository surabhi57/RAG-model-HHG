# Use the official Python image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements and install them
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the rest of your app code
COPY . /code

# Hugging Face Spaces use port 7860 by default
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]