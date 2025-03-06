# Contributing to Goliath Educational Platform

Thank you for your interest in contributing to the Goliath Educational Platform! This document provides guidelines and workflows for contributing to the project.

## Code of Conduct

- Be respectful and inclusive in all communications
- Provide constructive feedback
- Focus on the best outcome for the educational platform
- Consider the impact of changes on learners

## Development Workflow

### Getting Started

1. Set up your development environment following the instructions in DEVELOPMENT_GUIDE.md
2. Familiarize yourself with the architecture in ARCHITECTURE.md
3. Review the API documentation in API_DOCUMENTATION.md

### Making Changes

1. **Create an issue**: Describe the feature, bug, or improvement you're proposing
2. **Discussion**: Get feedback on your approach from maintainers
3. **Implementation**:
   - Create a feature branch from `main`
   - Make your changes
   - Add tests for new functionality
   - Update documentation as needed
4. **Code Review**: Submit a pull request for review
5. **Merge**: Once approved, your changes will be merged to `main`

## Coding Standards

### Style Guidelines

- Follow the Python style guide in DOCSTRING_TEMPLATE.md
- Adhere to the linting rules in .flake8
- Use the formatting rules in pyproject.toml (Black, isort)
- Run code formatting before committing:
  ```bash
  black .
  isort .
  flake8
  mypy
  ```

### Documentation

- Update documentation alongside code changes
- Add docstrings to all functions, classes, and modules
- Keep the README up to date
- Update API documentation when changing endpoints

### Testing

- Write tests for all new functionality
- Maintain or improve test coverage
- Include unit tests, integration tests, and end-to-end tests as appropriate
- Run tests before submitting pull requests:
  ```bash
  ./scripts/run-tests.sh
  ```

## Pull Request Guidelines

### PR Title Format

Use the following format:
```
[Component] Short description of change
```

Examples:
- `[Ptolemy] Add relationship filtering to concept API`
- `[Gutenberg] Fix template variable processing`

### PR Description Template

```
## Description
Brief description of the changes

## Related Issues
Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
Description of testing performed

## Screenshots (if appropriate)

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] API backwards compatibility maintained
```

## Microservice-Specific Guidelines

### Ptolemy (Knowledge Mapping)

- Maintain graph database integrity
- Document new relationship types
- Consider query performance

### Gutenberg (Content Generation)

- Follow template standards
- Document template variables
- Consider content quality and educational value

### Galileo (Learning Paths)

- Document learning path algorithms
- Consider performance with large datasets
- Test with diverse learner profiles

### Socrates (Learner Interaction)

- Prioritize user experience
- Consider accessibility
- Test with real educational scenarios

## Release Process

1. Version bumps follow semantic versioning
2. Release notes document all significant changes
3. Deployment follows CI/CD pipeline with staging environment testing
4. Monitoring of production deployment for issues

## Getting Help

- Ask questions in the #development channel
- Tag appropriate maintainers for specific components
- Refer to the documentation directory for more detailed guides

Thank you for contributing to education innovation!