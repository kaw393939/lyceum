#!/usr/bin/env python3

import os
import sys
import json
import click
from typing import Dict, List, Any, Optional

from thales.database.inspector import DatabaseInspector
from thales.database.consistency import ConsistencyVerifier
from thales.generators.concept_generator import MockDataGenerator
from thales.runners.integration_runner import IntegrationTestRunner


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Thales - Integration testing toolkit for Goliath Educational Platform."""
    pass


@main.command("run")
@click.option("--scenario", required=True, help="Test scenario to run")
@click.option("--config", default="config/test_scenarios.yaml", help="Test configuration file")
@click.option("--db-config", default="config/database_config.yaml", help="Database configuration file")
@click.option("--output-dir", default="reports", help="Directory to store test reports")
def run_cmd(scenario, config, db_config, output_dir):
    """Run integration test scenarios."""
    click.echo(f"Running test scenario: {scenario}")
    
    runner = IntegrationTestRunner(
        config_path=config,
        db_config_path=db_config,
        output_dir=output_dir
    )
    
    result = runner.run(scenario_name=scenario)
    
    if result["status"] == "success":
        click.echo(f"Test scenario '{scenario}' completed successfully")
    else:
        click.echo(f"Test scenario '{scenario}' failed: {result.get('message', 'Unknown error')}")
        sys.exit(1)


@main.command("verify")
@click.option("--source", default="mongodb", help="Source database type (mongodb or neo4j)")
@click.option("--target", default="neo4j", help="Target database type (mongodb or neo4j)")
@click.option("--entity-type", default="concepts", help="Entity type to verify (concepts, relationships, embeddings)")
@click.option("--config", default="config/database_config.yaml", help="Database configuration file")
@click.option("--output-dir", default="reports", help="Directory to store verification reports")
@click.option("--all", is_flag=True, help="Run all verification checks")
def verify_cmd(source, target, entity_type, config, output_dir, all):
    """Verify data consistency between different data stores."""
    click.echo(f"Verifying {entity_type} consistency between {source} and {target}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    inspector = DatabaseInspector(config_file=config)
    verifier = ConsistencyVerifier(inspector)
    
    if entity_type == "concepts" or all:
        result = verifier.verify_concepts(source_db=source, target_db=target)
        
        click.echo(f"Concepts in {source}: {result['total_mongodb_concepts']}")
        click.echo(f"Concepts in {target}: {result['total_neo4j_concepts']}")
        click.echo(f"Inconsistencies: {result['inconsistency_count']}")
        
        # Save report
        report_file = os.path.join(
            output_dir, 
            f"concept_verification_{source}_{target}.json"
        )
        with open(report_file, "w") as f:
            json.dump(result, f, default=str, indent=2)
            
        click.echo(f"Verification report saved to: {report_file}")
        
        if result["inconsistency_count"] > 0:
            click.echo("\nInconsistencies:")
            for i, inconsistency in enumerate(result["inconsistencies"][:5]):
                click.echo(f"  {i+1}. {inconsistency['id']}: {inconsistency['error']}")
                
            if len(result["inconsistencies"]) > 5:
                click.echo(f"  ... and {len(result['inconsistencies']) - 5} more")
    
    if entity_type == "relationships" or all:
        result = verifier.verify_relationships()
        
        click.echo("\nRelationships:")
        click.echo(f"Relationships in MongoDB: {result['total_mongodb_relationships']}")
        click.echo(f"Relationships in Neo4j: {result['total_neo4j_relationships']}")
        click.echo(f"Missing in Neo4j: {len(result['missing_in_neo4j'])}")
        click.echo(f"Missing in MongoDB: {len(result['missing_in_mongodb'])}")
        
        # Save report
        report_file = os.path.join(
            output_dir, 
            f"relationship_verification.json"
        )
        with open(report_file, "w") as f:
            json.dump(result, f, default=str, indent=2)
            
        click.echo(f"Verification report saved to: {report_file}")
    
    if entity_type == "embeddings" or all:
        result = verifier.verify_vector_embeddings()
        
        click.echo("\nVector Embeddings:")
        click.echo(f"Total concepts: {result['total_concepts']}")
        click.echo(f"Total vectors: {result['total_vectors']}")
        click.echo(f"Missing vectors: {len(result['missing_vectors'])}")
        click.echo(f"Consistency: {result['consistency_percentage']:.2f}%")
        
        # Save report
        report_file = os.path.join(
            output_dir, 
            f"embedding_verification.json"
        )
        with open(report_file, "w") as f:
            json.dump(result, f, default=str, indent=2)
            
        click.echo(f"Verification report saved to: {report_file}")


@main.command("generate")
@click.option("--concepts", default=10, help="Number of concepts to generate")
@click.option("--relationships", default=None, type=int, help="Relationships per concept")
@click.option("--output", help="Path to write generated concepts to")
@click.option("--populate-mongodb", is_flag=True, help="Populate MongoDB with generated concepts")
@click.option("--populate-neo4j", is_flag=True, help="Populate Neo4j with generated concepts")
@click.option("--populate-qdrant", is_flag=True, help="Populate Qdrant with generated embeddings")
@click.option("--populate-all", is_flag=True, help="Populate all databases")
@click.option("--config", default="config/database_config.yaml", help="Database configuration file")
def generate_cmd(concepts, relationships, output, populate_mongodb, populate_neo4j, populate_qdrant, populate_all, config):
    """Generate mock data for testing."""
    click.echo(f"Generating {concepts} test concepts")
    
    # If populate_all is specified, set all populate flags to True
    if populate_all:
        populate_mongodb = True
        populate_neo4j = True
        populate_qdrant = True
    
    # If we need to populate any database, create the necessary clients
    db_clients = {}
    
    if populate_mongodb or populate_neo4j or populate_qdrant:
        inspector = DatabaseInspector(config_file=config)
        
        if populate_mongodb:
            db_clients["mongodb"] = inspector.clients.get("mongodb")
            
        if populate_neo4j:
            db_clients["neo4j"] = inspector.clients.get("neo4j")
            
        if populate_qdrant:
            db_clients["qdrant"] = inspector.clients.get("qdrant")
    
    # Create generator and generate concepts
    generator = MockDataGenerator(db_clients=db_clients)
    
    # If we need to populate any database, use generate_and_populate_all
    if populate_mongodb or populate_neo4j or populate_qdrant:
        result = generator.generate_and_populate_all(
            concept_count=concepts,
            output_path=output
        )
        
        click.echo(f"Generated {result['concepts_generated']} concepts")
        
        if "mongodb" in result["databases_populated"]:
            click.echo(f"Populated MongoDB with {result['databases_populated']['mongodb']} concepts")
            
        if "neo4j" in result["databases_populated"]:
            click.echo(f"Populated Neo4j with {result['databases_populated']['neo4j']} concepts")
            
        if "qdrant" in result["databases_populated"]:
            click.echo(f"Populated Qdrant with {result['databases_populated']['qdrant']} vectors")
    else:
        # Otherwise, just generate concepts
        concepts_data = generator.generate_mock_concepts(
            count=concepts,
            output_path=output
        )
        
        click.echo(f"Generated {len(concepts_data)} concepts")
        if output:
            click.echo(f"Wrote concepts to {output}")


@main.command("inspect")
@click.argument("db_type", type=click.Choice(["mongodb", "neo4j", "qdrant"]))
@click.argument("database", required=False)
@click.argument("collection", required=False)
@click.option("--query", help="Query for MongoDB (JSON string)")
@click.option("--cypher", help="Cypher query for Neo4j")
@click.option("--limit", default=10, help="Maximum number of items to return")
@click.option("--config", default="config/database_config.yaml", help="Database configuration file")
@click.option("--output", help="Path to write output to")
def inspect_cmd(db_type, database, collection, query, cypher, limit, config, output):
    """Inspect database contents."""
    inspector = DatabaseInspector(config_file=config)
    
    if db_type == "mongodb":
        if not database or not collection:
            click.echo("Error: Both database and collection are required for MongoDB", err=True)
            sys.exit(1)
        
        query_obj = None
        if query:
            try:
                query_obj = json.loads(query)
            except json.JSONDecodeError:
                click.echo(f"Error: Invalid JSON query: {query}", err=True)
                sys.exit(1)
        
        result = inspector.inspect_mongodb(
            database=database,
            collection=collection,
            query=query_obj,
            limit=limit
        )
        
        if output:
            with open(output, "w") as f:
                f.write(result)
            click.echo(f"Output written to {output}")
        else:
            click.echo(result)
        
    elif db_type == "neo4j":
        if not cypher:
            click.echo("Error: Cypher query is required for Neo4j", err=True)
            sys.exit(1)
        
        result = inspector.inspect_neo4j(query=cypher)
        
        if output:
            with open(output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            click.echo(f"Output written to {output}")
        else:
            click.echo(json.dumps(result, indent=2, default=str))
        
    elif db_type == "qdrant":
        if not collection:
            click.echo("Error: Collection name is required for Qdrant", err=True)
            sys.exit(1)
        
        result = inspector.inspect_qdrant(collection=collection, limit=limit)
        
        click.echo(f"Retrieved {len(result)} vectors from Qdrant collection '{collection}'")
        
        if output:
            with open(output, "w") as f:
                json.dump([{"id": v.id, "payload": v.payload} for v in result], f, indent=2, default=str)
            click.echo(f"Output written to {output}")


@main.command("loadtest")
@click.option("--target", required=True, help="Target service (ptolemy, gutenberg, etc.)")
@click.option("--endpoint", required=True, help="API endpoint to test")
@click.option("--method", default="GET", help="HTTP method to use")
@click.option("--data", help="JSON request body for POST/PUT requests")
@click.option("--requests", default=100, help="Number of requests to send")
@click.option("--concurrency", default=10, help="Number of concurrent requests")
@click.option("--output", help="Path to write HTML report to")
def loadtest_cmd(target, endpoint, method, data, requests, concurrency, output):
    """Run load tests against services."""
    click.echo(f"Running load test against {target} {endpoint}")
    
    # First check that the target is available
    import httpx
    
    base_url = os.environ.get(f"{target.upper()}_URL", f"http://localhost:8000")
    health_endpoint = "/health"
    
    try:
        response = httpx.get(f"{base_url}{health_endpoint}")
        if response.status_code != 200:
            click.echo(f"Warning: {target} health check failed with status {response.status_code}")
    except Exception as e:
        click.echo(f"Warning: {target} health check failed: {str(e)}")
    
    # Import the load tester
    from thales.load.load_tester import LoadTester
    import asyncio
    
    tester = LoadTester(
        target_url=base_url,
        concurrency=concurrency,
        verbose=True
    )
    
    # Prepare request payload if provided
    payload = None
    if data:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            click.echo(f"Error: Invalid JSON data: {data}", err=True)
            sys.exit(1)
    
    # Run the load test
    click.echo(f"Sending {requests} {method} requests to {base_url}{endpoint} with concurrency {concurrency}")
    
    try:
        results = asyncio.run(tester.run_test(
            endpoint=endpoint,
            num_requests=requests,
            method=method,
            payload=payload
        ))
        
        # Display results
        click.echo("\nLoad Test Results:")
        click.echo(f"Total requests: {results['total_requests']}")
        click.echo(f"Successful requests: {results['successful_requests']}")
        click.echo(f"Failed requests: {results['failed_requests']}")
        click.echo(f"Success rate: {results['success_rate']*100:.2f}%")
        click.echo(f"Average response time: {results['average_response_time_ms']:.2f} ms")
        click.echo(f"Requests per second: {results['requests_per_second']:.2f}")
        
        # Generate HTML report if requested
        if output:
            html_report = tester.generate_report(
                title=f"Load Test: {target} {endpoint}",
                output_file=output
            )
            click.echo(f"HTML report saved to: {output}")
            
    except Exception as e:
        click.echo(f"Error running load test: {str(e)}", err=True)
        sys.exit(1)


@main.command("diagnose")
@click.option("--services", required=True, help="Comma-separated list of services to diagnose")
@click.option("--days", default=1, help="Number of days to look back for logs")
@click.option("--log-dir", default="/var/log", help="Directory containing service logs")
@click.option("--output-dir", default="reports", help="Directory to store diagnose reports")
def diagnose_cmd(services, days, log_dir, output_dir):
    """Collect and analyze error logs from services."""
    service_list = [s.strip() for s in services.split(",")]
    click.echo(f"Collecting logs for: {', '.join(service_list)}")
    
    # Import error collection/analysis tools
    from thales.diagnostics.error_collector import ErrorCollector
    from thales.diagnostics.error_analyzer import ErrorAnalyzer
    
    collector = ErrorCollector(log_dir=log_dir, report_dir=output_dir)
    analyzer = ErrorAnalyzer()
    
    # Collect logs from each service
    service_logs = {}
    total_errors = 0
    
    for service in service_list:
        logs = collector.collect_service_logs(service, days=days)
        service_logs[service] = logs
        total_errors += len(logs)
        click.echo(f"Collected {len(logs)} error logs from {service}")
    
    # Generate error report
    report_file = collector.generate_error_report(service_list, days=days)
    click.echo(f"Error report saved to: {report_file}")
    
    # If we have more than one service, analyze correlations
    if len(service_list) > 1:
        for i in range(len(service_list) - 1):
            for j in range(i + 1, len(service_list)):
                service1 = service_list[i]
                service2 = service_list[j]
                
                correlated_errors = analyzer.find_correlated_errors(
                    service_logs[service1],
                    service_logs[service2]
                )
                
                if correlated_errors:
                    click.echo(f"\nFound {len(correlated_errors)} correlated errors between {service1} and {service2}")
                    
                    # Save correlation report
                    correlation_file = os.path.join(
                        output_dir, 
                        f"correlation_{service1}_{service2}.json"
                    )
                    with open(correlation_file, "w") as f:
                        json.dump(correlated_errors, f, default=str, indent=2)
                        
                    click.echo(f"Correlation report saved to: {correlation_file}")


if __name__ == "__main__":
    main()