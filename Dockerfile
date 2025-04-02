# syntax=docker/dockerfile:1

# Use a slim Python base image
FROM python:3.9-slim AS base

# Set working directory
WORKDIR /app

# Install dependencies in a builder stage
FROM base AS builder

# Copy requirements file
COPY --link ./Lib/site-packages/requirements.txt ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /app/.venv && \
    /app/.venv/bin/pip install -r requirements.txt

# Final stage
FROM base AS final

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Set the PATH to include the virtual environment's bin directory
ENV PATH="/app/.venv/bin:$PATH"

# Copy the application code
COPY --link ./Lib/site-packages ./

# Set the entrypoint
CMD ["python", "app.py"]