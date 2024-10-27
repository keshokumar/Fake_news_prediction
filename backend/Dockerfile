# Use a base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files to the container
COPY ./backend/src/ ./src/  # Make sure to use relative path based on context
COPY ./backend/model/ ./model/  # Make sure to adjust this if needed
COPY ./backend/tokenizer.pickle ./  # Copy the tokenizer file

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["python", "src/app.py"]
