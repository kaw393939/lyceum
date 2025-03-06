# Goliath Platform Development Guide

This guide covers development workflows, best practices, and troubleshooting for the Goliath Educational Platform.

## Development Environment Setup

### Prerequisites
- Docker and Docker Compose
- Python 3.10+
- Git

### Initial Setup
1. Clone the repository
2. Create virtual environments for each service:
```bash
cd ptolemy && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && deactivate
cd ../gutenberg && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && deactivate
cd ../socrates && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && deactivate
cd ../galileo && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && deactivate
```
3. Start services in development mode:
```bash
./scripts/dev-start.sh
```

## Development Workflow

### Code Style and Quality
- Use the `.flake8` configuration
- Follow docstring standards in `DOCSTRING_TEMPLATE.md`
- Run linting before committing:
```bash
cd <service-directory>
flake8
```

### Making Changes
1. Create a feature branch
2. Implement changes
3. Add tests for new functionality
4. Verify all tests pass
5. Submit pull request

### Testing
- Unit tests: Test individual components in isolation
- Integration tests: Test communication between services
- End-to-end tests: Test complete user workflows

Run tests with:
```bash
# All tests
./scripts/run-tests.sh

# Service-specific tests
cd <service-directory> && pytest
```

## Database Management

### MongoDB
- Each service has its own database
- Use consistent naming with service prefixes
- Authentication details in environment variables

### Neo4j
- Used primarily by Ptolemy for knowledge graph
- Access via the Neo4j browser at http://localhost:7474

### Qdrant
- Used for vector storage and semantic search
- Monitor at http://localhost:6333/dashboard

## Troubleshooting

### Service Connectivity Issues
1. Check if all containers are running:
```bash
docker-compose ps
```
2. Check service logs:
```bash
docker-compose logs <service-name>
```
3. Restart a specific service:
```bash
docker-compose restart <service-name>
```

### Database Issues
If experiencing database connection issues:
```bash
./restart-database.sh
```

### API Debugging
- Use the `/debug` endpoint on any service for health checks
- Each service has a `/diagnostics` route for detailed status

## Deployment

### Development Environment
- Local Docker containers
- Mock LLM services

### Production Environment
- Container orchestration with Kubernetes
- Real LLM service integrations
- Monitoring and logging infrastructure

## Documentation

- API documentation is generated automatically
- Update the README.md for major architectural changes
- Document known issues in their respective fix files