import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta


class ErrorCollector:
    """Collects and aggregates errors from service logs and databases."""
    
    def __init__(self, log_dir: str = "/var/log", report_dir: str = "reports"):
        """Initialize the error collector.
        
        Args:
            log_dir: Directory containing service logs
            report_dir: Directory to store error reports
        """
        self.log_dir = log_dir
        self.report_dir = report_dir
        
        # Create report directory if it doesn't exist
        os.makedirs(report_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("error_collector")
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
        
    def collect_service_logs(
        self, 
        service_name: str, 
        days: int = 1, 
        level: str = "ERROR"
    ) -> List[Dict[str, Any]]:
        """Collects error logs from service log files.
        
        Args:
            service_name: Name of the service
            days: Number of days to look back
            level: Log level to filter by (ERROR, WARNING, INFO, etc.)
            
        Returns:
            List of error log entries
        """
        service_log_path = os.path.join(self.log_dir, service_name)
        logs = []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        self.logger.info(f"Collecting {level} logs for {service_name} from {start_date} to {end_date}")
        
        # Find relevant log files
        log_files = []
        if os.path.exists(service_log_path):
            for file in os.listdir(service_log_path):
                if file.endswith(".log"):
                    log_file_path = os.path.join(service_log_path, file)
                    log_files.append(log_file_path)
        
        self.logger.info(f"Found {len(log_files)} log files for {service_name}")
        
        # Extract errors from log files
        for log_file in log_files:
            try:
                file_logs = self._parse_log_file(log_file, start_date, end_date, level)
                logs.extend(file_logs)
                self.logger.info(f"Extracted {len(file_logs)} {level} entries from {log_file}")
            except Exception as e:
                self.logger.error(f"Error parsing log file {log_file}: {str(e)}")
            
        return logs
    
    def _parse_log_file(
        self, 
        log_file: str, 
        start_date: datetime, 
        end_date: datetime, 
        level: str
    ) -> List[Dict[str, Any]]:
        """Parses a log file and extracts relevant error entries.
        
        Args:
            log_file: Path to log file
            start_date: Start date for filtering logs
            end_date: End date for filtering logs
            level: Log level to filter by
            
        Returns:
            List of error log entries
        """
        entries = []
        
        # Define regex patterns for different log formats
        patterns = [
            # Pattern 1: 2025-03-05 14:22:15,123 - ERROR - Message
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (.+)",
            
            # Pattern 2: [2025-03-05T14:22:15.123Z] ERROR: Message
            r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)\] (\w+): (.+)",
            
            # Pattern 3: 2025/03/05 14:22:15 [ERROR] Message
            r"(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)"
        ]
        
        with open(log_file, "r") as f:
            for line in f:
                try:
                    # Try each pattern until one matches
                    for pattern in patterns:
                        match = re.match(pattern, line.strip())
                        if match:
                            timestamp_str, log_level, message = match.groups()
                            
                            # Check if level matches
                            if level != "ALL" and log_level != level:
                                continue
                                
                            # Parse timestamp based on format
                            try:
                                if "," in timestamp_str:
                                    # Format: 2025-03-05 14:22:15,123
                                    timestamp = datetime.strptime(
                                        timestamp_str, "%Y-%m-%d %H:%M:%S,%f"
                                    )
                                elif "T" in timestamp_str:
                                    # Format: 2025-03-05T14:22:15.123Z
                                    timestamp = datetime.strptime(
                                        timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ"
                                    )
                                else:
                                    # Format: 2025/03/05 14:22:15
                                    timestamp = datetime.strptime(
                                        timestamp_str, "%Y/%m/%d %H:%M:%S"
                                    )
                                    
                                # Check if timestamp is in range
                                if start_date <= timestamp <= end_date:
                                    entries.append({
                                        "timestamp": timestamp,
                                        "level": log_level,
                                        "message": message
                                    })
                            except ValueError:
                                # Skip lines with invalid timestamp format
                                pass
                                
                            # Break the pattern loop if we found a match
                            break
                except Exception:
                    # Skip malformed lines
                    pass
                    
        return entries
    
    def collect_database_errors(
        self, 
        db_inspector: Any, 
        days: int = 1
    ) -> List[Dict[str, Any]]:
        """Collects error records from database error logs collection.
        
        Args:
            db_inspector: DatabaseInspector instance
            days: Number of days to look back
            
        Returns:
            List of error database records
        """
        query = {
            "timestamp": {
                "$gte": datetime.now() - timedelta(days=days)
            },
            "level": "ERROR"
        }
        
        try:
            result_json = db_inspector.inspect_mongodb(
                database="logs", 
                collection="errors",
                query=query,
                limit=1000
            )
            return json.loads(result_json)
        except Exception as e:
            self.logger.error(f"Error collecting database errors: {str(e)}")
            return []
    
    def generate_error_report(
        self, 
        service_names: List[str], 
        days: int = 1
    ) -> str:
        """Generates a comprehensive error report for specified services.
        
        Args:
            service_names: List of service names
            days: Number of days to look back
            
        Returns:
            Path to the generated report file
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "period_days": days,
            "services": {}
        }
        
        for service in service_names:
            service_logs = self.collect_service_logs(service, days)
            report["services"][service] = {
                "log_errors": service_logs,
                "error_count": len(service_logs),
                "most_frequent": self._get_most_frequent_errors(service_logs)
            }
            
        # Write report to file
        report_file = os.path.join(
            self.report_dir, 
            f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_file, "w") as f:
            json.dump(report, f, default=str, indent=2)
            
        self.logger.info(f"Error report saved to: {report_file}")
        return report_file
    
    def _get_most_frequent_errors(
        self, 
        logs: List[Dict[str, Any]], 
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Identifies most frequent error patterns in logs.
        
        Args:
            logs: List of log entries
            top_n: Number of top error patterns to return
            
        Returns:
            List of most frequent error patterns with counts
        """
        # Simple approach: count occurrences of each message
        message_counts = {}
        
        for log in logs:
            msg = log.get("message", "")
            
            # Remove specific IDs or timestamps from message to group similar errors
            # This is a simple example - a more sophisticated approach might use regex
            
            # Remove UUIDs
            msg = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "<UUID>", msg)
            
            # Remove specific IDs
            msg = re.sub(r"ID: [0-9a-zA-Z]+", "ID: <ID>", msg)
            
            # Remove timestamps
            msg = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}", "<TIMESTAMP>", msg)
            
            # Remove IP addresses
            msg = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "<IP>", msg)
            
            message_counts[msg] = message_counts.get(msg, 0) + 1
            
        # Sort by count and return top N
        return sorted(
            [{"message": k, "count": v} for k, v in message_counts.items()],
            key=lambda x: x["count"],
            reverse=True
        )[:top_n]