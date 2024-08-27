# Start with a base image that has Python
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the src directory contents into the container at /app
COPY . /app

# Set the PYTHONPATH environment variable to include /app
ENV PYTHONPATH=/app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r prediction_model/requirements.txt

# Run pytest to execute the tests
RUN python -u -m pytest tests/

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the FastAPI app
CMD ["uvicorn", "prediction_model.app.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]