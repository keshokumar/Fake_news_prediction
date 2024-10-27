# Use a base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files to the container
COPY backend/src/ ./src/
COPY backend/model/ ./model/  # Adjust the path based on your actual structure
COPY backend/tokenizer.pickle ./  # If the tokenizer is in the backend directory

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "src/app.py"]

