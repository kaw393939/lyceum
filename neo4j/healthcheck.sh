#!/bin/bash
# Simple health check for Neo4j
wget -O /dev/null -q http://localhost:7474 > /dev/null 2>&1
exit $?