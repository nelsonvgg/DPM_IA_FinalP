# Running the Project with Docker

This section provides instructions to build and run the project using Docker.

## Prerequisites

- Ensure Docker and Docker Compose are installed on your system.
- Python version 3.9 is used in the Dockerfile.

## Build and Run Instructions

1. Clone the repository and navigate to the project directory.
2. Build the Docker image and start the services using Docker Compose:

   ```bash
   docker-compose up --build
   ```

3. The application will be accessible at the specified ports.

## Configuration

- Environment variables can be set in a `.env` file (uncomment the `env_file` line in the `docker-compose.yml` file).
- Modify the `Dockerfile` or `docker-compose.yml` as needed for custom configurations.

## Exposed Ports

- The application exposes ports as defined in the `docker-compose.yml` file. Ensure these ports are available on your host system.

For further details, refer to the project's documentation or contact the development team.