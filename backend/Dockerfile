# Use a base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files to the container
COPY . .

# Copy the model and tokenizer files to the appropriate locations in the container
COPY model/fake_news_detection_model_BILSTM.h5 /app/model/fake_news_detection_model_BILSTM.h5
COPY tokenizer.pickle /app/tokenizer.pickle

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "src/app.py"]
