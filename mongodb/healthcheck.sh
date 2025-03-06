#!/bin/bash
# MongoDB custom health check script

# Credentials
MONGO_USERNAME="admin"
MONGO_PASSWORD="password"
AUTH_DB="admin"

# Variables
MAX_RETRIES=5
RETRY_INTERVAL=3

# Function to perform health check
check_mongodb() {
    for ((i=1; i<=$MAX_RETRIES; i++)); do
        echo "Attempt $i of $MAX_RETRIES: Checking MongoDB health..."
        
        # Try to connect to MongoDB and run ping command
        result=$(mongosh --eval "db.adminCommand('ping')" -u "$MONGO_USERNAME" -p "$MONGO_PASSWORD" --authenticationDatabase "$AUTH_DB" --quiet)
        status=$?
        
        if [ $status -eq 0 ] && [[ "$result" == *"ok"* ]]; then
            echo "MongoDB is healthy!"
            return 0
        else
            echo "MongoDB is not ready yet. Retrying in $RETRY_INTERVAL seconds..."
            sleep $RETRY_INTERVAL
        fi
    done
    
    echo "Failed to verify MongoDB health after $MAX_RETRIES attempts"
    return 1
}

# Execute the check
check_mongodb
exit $?