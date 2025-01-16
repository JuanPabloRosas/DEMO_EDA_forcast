FROM python:3.10-slim

# Install essential system dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git build-essential cmake curl unixodbc unixodbc-dev gunicorn && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set timezone
ENV TZ=America/Monterrey
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Upgrade pip and install Python dependencies
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt

# Copy application files
COPY ./ /app
WORKDIR /app

# Expose Streamlit's default port
EXPOSE 8501

# Define entrypoint for Streamlit app
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--logger.level=info"]