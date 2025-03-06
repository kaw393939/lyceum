#!/usr/bin/env python3

import argparse
import asyncio
import sys
import os
import logging
import uvicorn
import threading
import time

# Add project root to Python path to import Thales package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.thales.simulators.service_simulator import ServiceSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def start_simulator(service_name, port, error_rate, delay_ms, config_file=None):
    """Start a service simulator in a separate thread."""
    simulator = ServiceSimulator(
        service_name=service_name,
        port=port,
        error_rate=error_rate,
        delay_ms=delay_ms,
        config_file=config_file
    )
    
    logger.info(f"Starting {service_name} simulator on port {port}")
    uvicorn.run(
        simulator.app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


def main():
    """Parse arguments and start service simulators."""
    parser = argparse.ArgumentParser(description="Start service simulators for Ptolemy and Gutenberg")
    
    parser.add_argument("--ptolemy-port", type=int, default=8000,
                        help="Port for Ptolemy simulator (default: 8000)")
    
    parser.add_argument("--gutenberg-port", type=int, default=8001,
                        help="Port for Gutenberg simulator (default: 8001)")
    
    parser.add_argument("--error-rate", type=float, default=0.0,
                        help="Error rate for simulators (0.0 to 1.0, default: 0.0)")
    
    parser.add_argument("--delay-ms", type=int, default=0,
                        help="Artificial delay in milliseconds (default: 0)")
    
    parser.add_argument("--config", type=str, default=None,
                        help="Path to simulator configuration file")
    
    parser.add_argument("--only", type=str, choices=["ptolemy", "gutenberg"],
                        help="Start only one simulator")
    
    args = parser.parse_args()
    
    # Start Ptolemy simulator
    if args.only is None or args.only == "ptolemy":
        ptolemy_thread = threading.Thread(
            target=start_simulator,
            args=("ptolemy", args.ptolemy_port, args.error_rate, args.delay_ms, args.config),
            daemon=True
        )
        ptolemy_thread.start()
        
    # Start Gutenberg simulator
    if args.only is None or args.only == "gutenberg":
        # If starting both, wait a bit for the first one to start
        if args.only is None:
            time.sleep(1)
            
        gutenberg_thread = threading.Thread(
            target=start_simulator,
            args=("gutenberg", args.gutenberg_port, args.error_rate, args.delay_ms, args.config),
            daemon=True
        )
        gutenberg_thread.start()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Shutting down simulators...")
        sys.exit(0)


if __name__ == "__main__":
    main()