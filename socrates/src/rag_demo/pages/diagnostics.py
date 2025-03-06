"""
Diagnostics Page
==============
Provides diagnostic information about the Goliath educational system services.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

def format_timestamp(timestamp):
    """Format ISO timestamp for display."""
    if not timestamp:
        return "Never"
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp

def render_status_indicator(status):
    """Render a colored status indicator."""
    if status == "healthy":
        return "🟢 Healthy"
    elif status == "unknown":
        return "🟡 Unknown"
    elif status.startswith("error"):
        return f"🔴 {status}"
    else:
        return f"🔴 {status}"

def show_diagnostics():
    """Display diagnostics page for system monitoring."""
    st.title("🔍 System Diagnostics")
    st.markdown("Monitor the health and performance of the Goliath educational platform components.")
    
    if "rag_service" not in st.session_state:
        st.error("RAG service not initialized. Please return to the main page.")
        return
        
    # Get diagnostic information
    diagnostics = st.session_state.rag_service.get_diagnostics()
    
    # System Overview
    st.header("System Overview")
    
    # Create columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Uptime", f"{diagnostics['uptime_seconds']:.1f} seconds")
    
    with col2:
        total_requests = diagnostics["request_stats"]["total_requests"]
        success_rate = 0
        if total_requests > 0:
            success_rate = (diagnostics["request_stats"]["successful_requests"] / total_requests) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        avg_response = diagnostics["request_stats"]["average_response_time"]
        st.metric("Avg Response Time", f"{avg_response:.2f}s")
    
    # Service Health
    st.header("Service Health")
    
    # Create a dataframe for services
    services_data = []
    for service_name, service_info in diagnostics["services"].items():
        services_data.append({
            "Service": service_name.capitalize(),
            "Status": render_status_indicator(service_info["health_status"]),
            "URL": service_info["url"],
            "Mock Mode": "Enabled" if service_info["mock_mode"] else "Disabled",
            "Last Checked": format_timestamp(service_info["last_checked"])
        })
    
    services_df = pd.DataFrame(services_data)
    st.table(services_df)
    
    # Performance Metrics
    st.header("Performance Metrics")
    
    # Create metrics for token usage
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Request Statistics")
        stats = diagnostics["request_stats"]
        
        # Create bar chart of request counts
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Successful", "Failed", "Total"],
            y=[stats["successful_requests"], stats["failed_requests"], stats["total_requests"]],
            marker_color=["green", "red", "blue"]
        ))
        fig.update_layout(title="Request Counts", xaxis_title="Status", yaxis_title="Count")
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Response Times")
        if stats["average_response_time"] > 0:
            # Create gauge for average response time
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=stats["average_response_time"],
                title={"text": "Average Response Time (s)"},
                gauge={
                    "axis": {"range": [0, max(5, stats["average_response_time"] * 2)]},
                    "bar": {"color": "blue"},
                    "steps": [
                        {"range": [0, 1], "color": "green"},
                        {"range": [1, 3], "color": "yellow"},
                        {"range": [3, 5], "color": "orange"},
                        {"range": [5, 10], "color": "red"}
                    ]
                }
            ))
            st.plotly_chart(fig)
        else:
            st.info("No response time data available yet.")
    
    # RAG Configuration
    st.header("RAG Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**LLM Settings**")
        st.json({
            "model": diagnostics["model"],
            "api_key_available": diagnostics["openai_api_key_available"]
        })
    
    with col2:
        st.write("**Retrieval Settings**")
        st.json({
            "top_k": st.session_state.rag_service.top_k,
            "temperature": st.session_state.rag_service.temperature
        })
    
    # Raw JSON Data (expandable)
    with st.expander("Raw Diagnostic Data"):
        st.json(diagnostics)
    
    # Refresh button
    if st.button("🔄 Refresh Diagnostics"):
        st.rerun()

# Main entrypoint for the diagnostics page
if __name__ == "__main__":
    show_diagnostics()