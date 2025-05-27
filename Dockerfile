# Use an official Python runtime as a parent image
FROM python:3.12-slim

RUN apt-get update && apt-get install -y libgl1

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run app.py when the container launches
CMD ["python", "CameraAI-1.4/app.py"]