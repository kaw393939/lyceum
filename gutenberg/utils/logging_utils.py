"""
Gutenberg Content Generation System - Logging Utilities
======================================================
Utilities for logging and log analysis.
"""

import os
import json
import logging
import time
import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Pattern, Callable
from logging.handlers import RotatingFileHandler
import threading

# Try to import structlog for structured logging
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

# Configure thread-local storage for request IDs
thread_local = threading.local()

class RequestIdFilter(logging.Filter):
    """
    Filter that adds request_id to log records.
    This allows tracking of logs across a single request.
    """
    def filter(self, record):
        # Get request ID from thread-local storage or use None
        if hasattr(thread_local, 'request_id'):
            record.request_id = thread_local.request_id
        else:
            record.request_id = 'no_request_id'
        return True

class ThreadRequestIdLogger(logging.Logger):
    """
    Custom logger subclass that supports request ID tracking.
    """
    def set_request_id(self, request_id):
        """Set the request ID for the current thread."""
        thread_local.request_id = request_id
    
    def clear_request_id(self):
        """Clear the request ID from the current thread."""
        if hasattr(thread_local, 'request_id'):
            del thread_local.request_id

def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = "gutenberg.log",
    console: bool = True,
    use_structured_logging: bool = False,
    max_file_size: int = 10*1024*1024,  # 10 MB
    backup_count: int = 5
) -> None:
    """
    Configure the logging system.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None to disable file logging)
        console: Whether to log to console
        use_structured_logging: Whether to use structured logging (requires structlog)
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
    """
    # Convert string level to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatters
    standard_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s'
    )
    
    # Create handlers
    handlers = []
    
    # Add file handler if configured
    if log_file:
        try:
            # Create directory for log file if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            # Create rotating file handler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setFormatter(standard_formatter)
            handlers.append(file_handler)
        except Exception as e:
            print(f"Error setting up file logging: {e}")
    
    # Add console handler if configured
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(standard_formatter)
        handlers.append(console_handler)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    
    # Add request ID filter to all handlers
    request_id_filter = RequestIdFilter()
    for handler in handlers:
        handler.addFilter(request_id_filter)
        root_logger.addHandler(handler)
    
    # Register custom logger class
    logging.setLoggerClass(ThreadRequestIdLogger)
    
    # Configure structured logging if requested and available
    if use_structured_logging and STRUCTLOG_AVAILABLE:
        try:
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
        except Exception as e:
            print(f"Error setting up structured logging: {e}")
    
    # Silence some noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: The name of the logger
        
    Returns:
        A logger instance
    """
    return logging.getLogger(name)

class LogAnalyzer:
    """
    Log analyzer for detecting issues and generating reports.
    """
    
    def __init__(
        self, 
        log_file: str = "gutenberg.log",
        analysis_interval: int = 3600,  # 1 hour
        auto_analyze: bool = False
    ):
        """
        Initialize the log analyzer.
        
        Args:
            log_file: Path to the log file to analyze
            analysis_interval: Interval for automatic analysis in seconds
            auto_analyze: Whether to start automatic analysis
        """
        self.log_file = log_file
        self.analysis_interval = analysis_interval
        self._auto_analyze = False
        self._auto_analyze_thread = None
        self._stop_event = threading.Event()
        self.logger = get_logger("log_analyzer")
        
        # Patterns for log analysis
        self.error_pattern = re.compile(r'ERROR|CRITICAL|Exception|Error|Failed|Traceback')
        self.slow_op_pattern = re.compile(r'took (\d+(?:\.\d+)?)s')
        self.api_pattern = re.compile(r'(GET|POST|PUT|DELETE|PATCH) (/?[a-zA-Z0-9_/]+)')
        
        # Initialize counters
        self.reset_counters()
        
        # Start auto-analysis if requested
        if auto_analyze:
            self.start_auto_analysis()
    
    def reset_counters(self) -> None:
        """Reset all counters for a new analysis."""
        self.error_count = 0
        self.slow_op_count = 0
        self.api_call_count = 0
        self.error_types = {}
        self.slow_operations = {}
        self.api_endpoints = {}
        self.request_ids = set()
        self.requests_with_errors = set()
        self.slow_requests = set()
    
    def analyze(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Analyze the log file and generate a report.
        
        Args:
            since: Only analyze log entries since this time (None for all)
            
        Returns:
            A dictionary containing analysis results
        """
        self.logger.info(f"Starting log analysis of {self.log_file}")
        start_time = time.time()
        self.reset_counters()
        
        if not os.path.exists(self.log_file):
            self.logger.error(f"Log file {self.log_file} not found")
            return self._generate_error_report("Log file not found")
        
        try:
            # Convert since to timestamp string for comparison
            since_str = None
            if since:
                since_str = since.strftime("%Y-%m-%d %H:%M:%S")
            
            # Process each line in the log file
            with open(self.log_file, 'r') as f:
                for line in f:
                    # Check if line is after since time
                    if since_str:
                        # Extract timestamp (assumes standard format)
                        timestamp_match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                        if timestamp_match:
                            line_time = timestamp_match.group(1)
                            if line_time < since_str:
                                continue
                    
                    # Extract request ID
                    request_id_match = re.search(r'\[([\w-]+)\]', line)
                    request_id = request_id_match.group(1) if request_id_match else None
                    
                    if request_id and request_id != "no_request_id":
                        self.request_ids.add(request_id)
                    
                    # Check for errors
                    if self.error_pattern.search(line):
                        self.error_count += 1
                        
                        # Extract error type
                        error_type = "Unknown Error"
                        error_match = re.search(r'ERROR.*?- (.*?)(?:\s*\[|\s*$)', line)
                        if error_match:
                            error_type = error_match.group(1).strip()
                        
                        # Update error counters
                        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
                        
                        # Track request IDs with errors
                        if request_id and request_id != "no_request_id":
                            self.requests_with_errors.add(request_id)
                    
                    # Check for slow operations
                    slow_match = self.slow_op_pattern.search(line)
                    if slow_match:
                        duration = float(slow_match.group(1))
                        if duration > 1.0:  # Consider operations taking over 1s as slow
                            self.slow_op_count += 1
                            
                            # Extract operation name
                            operation = "Unknown Operation"
                            op_match = re.search(r'- (.*?) took', line)
                            if op_match:
                                operation = op_match.group(1).strip()
                            
                            # Update slow operation counters
                            if operation in self.slow_operations:
                                self.slow_operations[operation]["count"] += 1
                                self.slow_operations[operation]["total_time"] += duration
                                if duration > self.slow_operations[operation]["max_time"]:
                                    self.slow_operations[operation]["max_time"] = duration
                            else:
                                self.slow_operations[operation] = {
                                    "count": 1,
                                    "total_time": duration,
                                    "max_time": duration
                                }
                            
                            # Track request IDs with slow operations
                            if request_id and request_id != "no_request_id":
                                self.slow_requests.add(request_id)
                    
                    # Check for API calls
                    api_match = self.api_pattern.search(line)
                    if api_match:
                        method, endpoint = api_match.groups()
                        self.api_call_count += 1
                        
                        # Update API endpoint counters
                        key = f"{method} {endpoint}"
                        self.api_endpoints[key] = self.api_endpoints.get(key, 0) + 1
            
            # Generate the report
            report = self._generate_report()
            self.logger.info(f"Log analysis completed in {time.time() - start_time:.2f}s")
            return report
            
        except Exception as e:
            self.logger.error(f"Error analyzing log file: {e}")
            return self._generate_error_report(str(e))
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate a report from the analysis data."""
        # Calculate averages for slow operations
        slow_ops_avg = []
        for op, data in self.slow_operations.items():
            avg_time = data["total_time"] / data["count"]
            slow_ops_avg.append({
                "operation": op,
                "count": data["count"],
                "avg_time": avg_time,
                "max_time": data["max_time"]
            })
        # Sort by average time (descending)
        slow_ops_avg.sort(key=lambda x: x["avg_time"], reverse=True)
        
        # Most common errors
        error_items = list(self.error_types.items())
        error_items.sort(key=lambda x: x[1], reverse=True)
        most_common_errors = [{"type": error, "count": count} for error, count in error_items]
        
        # Most used API endpoints
        endpoint_items = list(self.api_endpoints.items())
        endpoint_items.sort(key=lambda x: x[1], reverse=True)
        most_used_endpoints = [{"endpoint": endpoint, "count": count} for endpoint, count in endpoint_items]
        
        # Calculate error rate
        total_requests = len(self.request_ids)
        error_rate = len(self.requests_with_errors) / max(1, total_requests)
        slow_rate = len(self.slow_requests) / max(1, total_requests)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(error_rate, slow_rate)
        
        # Create the report
        report = {
            "analysis_time": datetime.now().isoformat(),
            "log_file": self.log_file,
            "summary": {
                "total_requests": total_requests,
                "requests_with_errors": len(self.requests_with_errors),
                "requests_with_slow_ops": len(self.slow_requests),
                "error_rate": error_rate,
                "slow_operation_rate": slow_rate
            },
            "error_summary": {
                "total_errors": self.error_count,
                "unique_error_types": len(self.error_types),
                "most_common_errors": most_common_errors
            },
            "performance_summary": {
                "slow_operations_count": self.slow_op_count,
                "slowest_operations": slow_ops_avg
            },
            "api_summary": {
                "total_api_calls": self.api_call_count,
                "unique_endpoints": len(self.api_endpoints),
                "most_used_endpoints": most_used_endpoints
            },
            "recommendations": recommendations
        }
        
        return report
    
    def _generate_error_report(self, error_message: str) -> Dict[str, Any]:
        """Generate an error report when analysis fails."""
        return {
            "analysis_time": datetime.now().isoformat(),
            "log_file": self.log_file,
            "error": error_message,
            "success": False
        }
    
    def _generate_recommendations(self, error_rate: float, slow_rate: float) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Error rate recommendations
        if error_rate > 0.1:
            recommendations.append(f"High error rate ({error_rate:.1%}). Investigate most common errors.")
        
        # Add specific recommendations for top errors
        for i, error in enumerate(list(self.error_types.items())[:3]):
            error_type, count = error
            if count > 5:
                recommendations.append(f"Frequent error '{error_type}' occurred {count} times.")
        
        # Slow operation recommendations
        if slow_rate > 0.1:
            recommendations.append(f"High rate of slow operations ({slow_rate:.1%}). Review performance bottlenecks.")
        
        # Add specific recommendations for slowest operations
        slow_ops = sorted(self.slow_operations.items(), key=lambda x: x[1]["avg_time"], reverse=True)
        for i, (op, data) in enumerate(slow_ops[:3]):
            if data["avg_time"] > 3.0:  # Operations taking more than 3 seconds on average
                recommendations.append(
                    f"Slow operation '{op}' takes {data['avg_time']:.2f}s on average. Consider optimization."
                )
        
        # API usage recommendations
        top_endpoints = sorted(self.api_endpoints.items(), key=lambda x: x[1], reverse=True)
        if top_endpoints and top_endpoints[0][1] > 100:
            endpoint, count = top_endpoints[0]
            recommendations.append(f"Heavy usage of endpoint '{endpoint}' ({count} calls). Consider caching or rate limiting.")
        
        # Add general recommendations if none specific
        if not recommendations:
            if self.error_count > 0:
                recommendations.append("Review error logs for potential issues.")
            if self.slow_op_count > 0:
                recommendations.append("Monitor slow operations for performance optimization opportunities.")
            if not self.error_count and not self.slow_op_count:
                recommendations.append("No significant issues detected. Continue monitoring.")
        
        return recommendations
    
    def get_latest_report(self) -> Dict[str, Any]:
        """Get the latest analysis report or generate one if needed."""
        return self.analyze()
    
    def save_report(self, report: Dict[str, Any], filename: str) -> bool:
        """
        Save a report to a file.
        
        Args:
            report: The report to save
            filename: The filename to save to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Saved analysis report to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")
            return False
    
    def _auto_analysis_worker(self) -> None:
        """Worker function for automatic analysis."""
        self.logger.info(f"Starting automatic log analysis (interval: {self.analysis_interval}s)")
        
        while not self._stop_event.is_set():
            try:
                # Sleep for the specified interval, checking stop event periodically
                for _ in range(int(self.analysis_interval / 5)):
                    if self._stop_event.is_set():
                        break
                    time.sleep(min(5, self.analysis_interval))
                
                if self._stop_event.is_set():
                    break
                
                # Run analysis
                self.logger.info("Running scheduled log analysis")
                since = datetime.now() - timedelta(seconds=self.analysis_interval)
                report = self.analyze(since=since)
                
                # Log a summary of findings
                if "error" not in report:
                    error_count = report["error_summary"]["total_errors"]
                    slow_count = report["performance_summary"]["slow_operations_count"]
                    self.logger.info(f"Analysis complete: found {error_count} errors and {slow_count} slow operations")
                    
                    # Log recommendations
                    for rec in report["recommendations"]:
                        self.logger.info(f"Recommendation: {rec}")
                else:
                    self.logger.error(f"Analysis failed: {report.get('error', 'Unknown error')}")
                
            except Exception as e:
                self.logger.error(f"Error in auto-analysis worker: {e}")
    
    def start_auto_analysis(self) -> bool:
        """
        Start automatic periodic analysis.
        
        Returns:
            True if started, False if already running
        """
        if self._auto_analyze:
            return False
        
        self._auto_analyze = True
        self._stop_event.clear()
        self._auto_analyze_thread = threading.Thread(
            target=self._auto_analysis_worker,
            daemon=True
        )
        self._auto_analyze_thread.start()
        self.logger.info("Automatic log analysis started")
        return True
    
    def stop_auto_analysis(self) -> bool:
        """
        Stop automatic periodic analysis.
        
        Returns:
            True if stopped, False if not running
        """
        if not self._auto_analyze:
            return False
        
        self._auto_analyze = False
        self._stop_event.set()
        if self._auto_analyze_thread:
            self._auto_analyze_thread.join(timeout=5)
        self.logger.info("Automatic log analysis stopped")
        return True

def create_log_analyzer(
    log_file: str = "gutenberg.log",
    analysis_interval: int = 3600,
    auto_analyze: bool = False
) -> LogAnalyzer:
    """
    Create a log analyzer instance.
    
    Args:
        log_file: Path to the log file to analyze
        analysis_interval: Interval for automatic analysis in seconds
        auto_analyze: Whether to start automatic analysis
        
    Returns:
        A LogAnalyzer instance
    """
    return LogAnalyzer(
        log_file=log_file,
        analysis_interval=analysis_interval,
        auto_analyze=auto_analyze
    )