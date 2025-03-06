#!/bin/bash
# Simple healthcheck for Qdrant
ENDPOINT="http://localhost:6333/readiness"
wget -q -O - $ENDPOINT > /dev/null 2>&1
exit $?