# Goliath Platform API Documentation

This document provides standardized API documentation for the Goliath Educational Platform microservices.

## API Standards

All APIs across the Goliath platform follow these standards:

- REST-based architecture
- JSON request and response format
- Authorization via API keys in headers
- Consistent error response format
- Versioned endpoints (/v1/, /v2/, etc.)

### Standard Error Response

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field_name": "Specific error about this field"
    }
  },
  "request_id": "unique-request-id-for-tracing"
}
```

### Common Status Codes

- 200: Success
- 201: Created (new resource)
- 400: Bad Request (client error)
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 500: Internal Server Error

## Ptolemy API (Knowledge Mapping Service)

Base URL: `http://localhost:8000/api/v1`

### Concepts

#### Get All Concepts

```
GET /concepts
```

Parameters:
- `limit` (optional): Maximum number of results (default 100)
- `offset` (optional): Pagination offset (default 0)

Response:
```json
{
  "data": [
    {
      "id": "concept-id",
      "name": "Concept Name",
      "description": "Concept description",
      "metadata": {}
    }
  ],
  "pagination": {
    "total": 100,
    "limit": 20,
    "offset": 0
  }
}
```

#### Get Concept by ID

```
GET /concepts/{concept_id}
```

Response:
```json
{
  "id": "concept-id",
  "name": "Concept Name",
  "description": "Concept description",
  "metadata": {},
  "relationships": [
    {
      "type": "PREREQUISITE",
      "target_concept_id": "other-concept-id",
      "metadata": {}
    }
  ]
}
```

#### Create Concept

```
POST /concepts
```

Request:
```json
{
  "name": "New Concept",
  "description": "Concept description",
  "metadata": {}
}
```

Response:
```json
{
  "id": "new-concept-id",
  "name": "New Concept",
  "description": "Concept description",
  "metadata": {}
}
```

### Relationships

#### Create Relationship

```
POST /concepts/{concept_id}/relationships
```

Request:
```json
{
  "type": "PREREQUISITE",
  "target_concept_id": "other-concept-id",
  "metadata": {}
}
```

Response:
```json
{
  "id": "relationship-id",
  "type": "PREREQUISITE",
  "source_concept_id": "concept-id",
  "target_concept_id": "other-concept-id",
  "metadata": {}
}
```

### Vector Search

```
POST /vector-search
```

Request:
```json
{
  "query": "search query text",
  "limit": 10
}
```

Response:
```json
{
  "results": [
    {
      "concept_id": "concept-id",
      "name": "Concept Name",
      "similarity": 0.95
    }
  ]
}
```

## Gutenberg API (Content Generation Service)

Base URL: `http://localhost:8001/api/v1`

### Templates

#### Get All Templates

```
GET /templates
```

Parameters:
- `limit` (optional): Maximum number of results (default 100)
- `offset` (optional): Pagination offset (default 0)

Response:
```json
{
  "data": [
    {
      "id": "template-id",
      "name": "Template Name",
      "description": "Template description",
      "structure": {}
    }
  ],
  "pagination": {
    "total": 50,
    "limit": 20,
    "offset": 0
  }
}
```

#### Get Template by ID

```
GET /templates/{template_id}
```

Response:
```json
{
  "id": "template-id",
  "name": "Template Name",
  "description": "Template description",
  "structure": {},
  "created_at": "2025-01-01T00:00:00Z",
  "updated_at": "2025-01-02T00:00:00Z"
}
```

### Content Generation

#### Generate Content

```
POST /generate
```

Request:
```json
{
  "template_id": "template-id",
  "concept_id": "concept-id",
  "parameters": {
    "difficulty": "intermediate",
    "audience": "adult"
  }
}
```

Response:
```json
{
  "content_id": "content-id",
  "template_id": "template-id",
  "concept_id": "concept-id",
  "content": {},
  "media_ids": [
    "media-id-1"
  ]
}
```

### Media

#### Get Media

```
GET /media/{media_id}
```

Response: Binary file stream with appropriate Content-Type header

## Galileo API (Learning Path Recommendation)

Base URL: `http://localhost:8002/api/v1`

### Learners

#### Get Learner Profile

```
GET /learners/{learner_id}
```

Response:
```json
{
  "id": "learner-id",
  "name": "Learner Name",
  "preferences": {},
  "progress": {
    "concept-id": {
      "status": "completed",
      "score": 0.95,
      "last_activity": "2025-01-01T00:00:00Z"
    }
  }
}
```

### Learning Paths

#### Generate Learning Path

```
POST /learning-paths
```

Request:
```json
{
  "learner_id": "learner-id",
  "target_concept_id": "concept-id",
  "constraints": {
    "max_length": 10,
    "include_concepts": ["concept-id-1"],
    "exclude_concepts": ["concept-id-2"]
  }
}
```

Response:
```json
{
  "path_id": "path-id",
  "learner_id": "learner-id",
  "target_concept_id": "concept-id",
  "path": [
    {
      "concept_id": "concept-id-1",
      "order": 1,
      "estimated_time_minutes": 30
    },
    {
      "concept_id": "concept-id-3",
      "order": 2,
      "estimated_time_minutes": 45
    }
  ]
}
```

## Socrates API (Learner Interaction)

Base URL: `http://localhost:8003/api/v1`

### Chat

#### Send Message

```
POST /chat
```

Request:
```json
{
  "learner_id": "learner-id",
  "message": "User message text",
  "context": {
    "current_concept_id": "concept-id",
    "current_activity": "exercise"
  }
}
```

Response:
```json
{
  "message_id": "message-id",
  "response": "Assistant response text",
  "suggested_actions": [
    {
      "type": "VIEW_CONCEPT",
      "concept_id": "concept-id",
      "label": "Learn more about X"
    }
  ]
}
```

### Progress

#### Get Learner Progress

```
GET /learners/{learner_id}/progress
```

Response:
```json
{
  "overall_progress": 0.65,
  "concepts_completed": 15,
  "concepts_in_progress": 3,
  "concepts_by_status": {
    "not_started": 32,
    "in_progress": 3,
    "completed": 15
  },
  "recent_activity": [
    {
      "concept_id": "concept-id",
      "activity_type": "quiz",
      "timestamp": "2025-01-01T00:00:00Z",
      "result": "passed"
    }
  ]
}
```

## Authentication & Authorization

All API requests should include:

```
Authorization: Bearer <api_key>
```

Service-to-service communication uses internal API keys configured in environment variables.

## Rate Limiting

All endpoints are rate-limited based on:
- API key
- IP address
- Endpoint

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1614556800
```

## Webhook Notifications

Each service can send webhook notifications for important events.

Configure webhooks at:
```
POST /webhooks
```

Request:
```json
{
  "url": "https://your-service.com/webhook",
  "events": ["content.created", "learner.progress.updated"],
  "secret": "webhook-secret-for-signature-verification"
}
```

Response:
```json
{
  "webhook_id": "webhook-id",
  "url": "https://your-service.com/webhook",
  "events": ["content.created", "learner.progress.updated"]
}
```