# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install build tools and dependencies in a single layer to save space
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*


# Copy the Python script and data folders into the container
COPY text_classifier.py ./
COPY pos/ ./pos/
COPY neg/ ./neg/
COPY requirements.txt .
COPY . .


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm \
    && python -m nltk.downloader stopwords punkt punkt_tab 

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World


# Run nlp.py when the container launches
CMD ["python", "text_classifier.py"]