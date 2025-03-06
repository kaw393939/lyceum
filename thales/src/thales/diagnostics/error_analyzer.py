import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta


class ErrorAnalyzer:
    """Analyzes error patterns and correlations across services."""
    
    def __init__(self, time_window_seconds: int = 60):
        """Initialize the error analyzer.
        
        Args:
            time_window_seconds: Time window in seconds for correlating errors
        """
        self.time_window_seconds = time_window_seconds
        
        # Set up logging
        self.logger = logging.getLogger("error_analyzer")
        self.logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(console_handler)
        
    def find_correlated_errors(
        self, 
        service1_logs: List[Dict[str, Any]], 
        service2_logs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find correlated errors between two services.
        
        Args:
            service1_logs: List of error logs from service 1
            service2_logs: List of error logs from service 2
            
        Returns:
            List of correlated error patterns
        """
        # Sort logs by timestamp
        service1_logs = sorted(service1_logs, key=lambda x: x.get("timestamp"))
        service2_logs = sorted(service2_logs, key=lambda x: x.get("timestamp"))
        
        correlated_errors = []
        
        # Find errors in service1 that are followed by errors in service2 within the time window
        for log1 in service1_logs:
            timestamp1 = log1.get("timestamp")
            if not timestamp1:
                continue
                
            # Convert string timestamp to datetime if needed
            if isinstance(timestamp1, str):
                try:
                    timestamp1 = datetime.fromisoformat(timestamp1.replace("Z", "+00:00"))
                except ValueError:
                    continue
                    
            # Calculate end of time window
            window_end = timestamp1 + timedelta(seconds=self.time_window_seconds)
            
            # Find errors in service2 that occurred within the time window
            correlated_logs = []
            for log2 in service2_logs:
                timestamp2 = log2.get("timestamp")
                if not timestamp2:
                    continue
                    
                # Convert string timestamp to datetime if needed
                if isinstance(timestamp2, str):
                    try:
                        timestamp2 = datetime.fromisoformat(timestamp2.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                        
                # Check if error occurred within time window
                if timestamp1 <= timestamp2 <= window_end:
                    correlated_logs.append(log2)
                    
            # If we found correlated errors, add them to the result
            if correlated_logs:
                correlated_errors.append({
                    "service1_error": log1,
                    "service2_errors": correlated_logs,
                    "time_delta_seconds": [(log2.get("timestamp") - timestamp1).total_seconds() 
                                          for log2 in correlated_logs]
                })
                
        return correlated_errors
    
    def identify_common_patterns(
        self, 
        logs: List[Dict[str, Any]], 
        min_occurrences: int = 2
    ) -> List[Dict[str, Any]]:
        """Identify common error patterns in logs.
        
        Args:
            logs: List of error logs
            min_occurrences: Minimum number of occurrences to consider a pattern common
            
        Returns:
            List of common error patterns
        """
        # Extract error messages
        messages = [log.get("message", "") for log in logs]
        
        # Find common word sequences
        patterns = self._find_common_sequences(messages)
        
        # Filter patterns by minimum occurrences
        common_patterns = []
        for pattern, occurrences in patterns.items():
            if len(occurrences) >= min_occurrences:
                common_patterns.append({
                    "pattern": pattern,
                    "occurrences": len(occurrences),
                    "examples": occurrences[:5]  # Include up to 5 examples
                })
                
        # Sort by number of occurrences (descending)
        return sorted(common_patterns, key=lambda x: x["occurrences"], reverse=True)
    
    def _find_common_sequences(
        self, 
        messages: List[str], 
        min_length: int = 3
    ) -> Dict[str, List[str]]:
        """Find common word sequences in messages.
        
        Args:
            messages: List of error messages
            min_length: Minimum sequence length to consider
            
        Returns:
            Dictionary mapping sequences to messages containing them
        """
        sequences = {}
        
        # Process each message
        for message in messages:
            # Split message into words
            words = message.split()
            
            # Skip short messages
            if len(words) < min_length:
                continue
                
            # Generate all possible sequences of at least min_length words
            for i in range(len(words) - min_length + 1):
                for j in range(i + min_length, len(words) + 1):
                    sequence = " ".join(words[i:j])
                    
                    # Add to dictionary
                    if sequence not in sequences:
                        sequences[sequence] = []
                    sequences[sequence].append(message)
                    
        return sequences
    
    def classify_errors(
        self, 
        logs: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Classify errors into categories.
        
        Args:
            logs: List of error logs
            
        Returns:
            Dictionary mapping error categories to matching logs
        """
        categories = {
            "network": [],
            "database": [],
            "authentication": [],
            "timeout": [],
            "validation": [],
            "system": [],
            "other": []
        }
        
        # Keywords for each category
        keywords = {
            "network": ["connection", "network", "dns", "socket", "http", "tcp", "ip", "unreachable"],
            "database": ["database", "query", "mongodb", "neo4j", "qdrant", "collection", "document", "index"],
            "authentication": ["auth", "permission", "access", "denied", "unauthorized", "forbidden", "credentials"],
            "timeout": ["timeout", "timed out", "deadline", "expired"],
            "validation": ["validation", "invalid", "schema", "format", "required field", "constraint"],
            "system": ["memory", "disk", "cpu", "system", "os", "resource", "limit"]
        }
        
        # Classify each log
        for log in logs:
            message = log.get("message", "").lower()
            
            # Check each category
            categorized = False
            for category, words in keywords.items():
                if any(word in message for word in words):
                    categories[category].append(log)
                    categorized = True
                    break
                    
            # If not categorized, add to "other"
            if not categorized:
                categories["other"].append(log)
                
        return categories
    
    def generate_failure_timeline(
        self, 
        logs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate a timeline of errors to visualize failure propagation.
        
        Args:
            logs: List of error logs from multiple services
            
        Returns:
            List of timeline entries
        """
        # Sort logs by timestamp
        sorted_logs = sorted(logs, key=lambda x: x.get("timestamp"))
        
        # Group logs by minute for a clearer timeline
        timeline = {}
        
        for log in sorted_logs:
            timestamp = log.get("timestamp")
            service = log.get("service", "unknown")
            
            # Convert string timestamp to datetime if needed
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    continue
                    
            # Round to minute
            minute = timestamp.replace(second=0, microsecond=0)
            
            # Create timeline entry
            key = minute.isoformat()
            if key not in timeline:
                timeline[key] = {
                    "timestamp": minute,
                    "services": {}
                }
                
            # Add log to service
            if service not in timeline[key]["services"]:
                timeline[key]["services"][service] = []
                
            timeline[key]["services"][service].append(log)
            
        # Convert to list and sort by timestamp
        result = list(timeline.values())
        result.sort(key=lambda x: x["timestamp"])
        
        return result