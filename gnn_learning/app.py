#!/usr/bin/env python
"""
Streamlit Educational Achievement and Recommendation System

This application provides a user-friendly interface for the GNN-Based Educational
Achievement and Recommendation System.

Features:
- Manage learners, concepts, questions, and resources
- View learning statistics and progress
- Get personalized recommendations for learning paths
- Interact with intelligent tutoring through Q&A sessions
- Train and evaluate the GNN model
- Import and export educational content

Usage: streamlit run app.py
"""

import streamlit as st
import json
import os
import yaml
import base64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import uuid
import time
import tempfile
import sys
from pathlib import Path

# Import the educational system module
# Add import for gnn_education_system.py (the file containing the EducationalSystem class)
# Assuming you've renamed the original script to gnn_education_system.py
try:
    from gnn_education_system import EducationalSystem
except ImportError:
    st.error("Please ensure the GNN Educational System module is available (gnn_education_system.py)")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Educational Achievement System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'system' not in st.session_state:
    st.session_state.system = None
if 'current_learner' not in st.session_state:
    st.session_state.current_learner = None
if 'current_concept' not in st.session_state:
    st.session_state.current_concept = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# UI theme and styling
primary_color = "#4CAF50"
secondary_color = "#2196F3"
accent_color = "#FF9800"
danger_color = "#F44336"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #2196F3;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .stButton>button {
        width: 100%;
    }
    .concept-card {
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .stat-card {
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        text-align: center;
    }
    .stat-value {
        font-size: 24px;
        font-weight: bold;
        color: #2196F3;
    }
    .stat-label {
        font-size: 14px;
        color: #757575;
    }
    .achievement-card {
        border: 1px solid #FFD700;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #FFFDE7;
    }
    div[data-testid="stExpander"] details summary p {
        font-size: 1.1em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

#--------------------------
# Utility Functions
#--------------------------

def initialize_system():
    """Initialize the educational system"""
    with st.spinner("Initializing Educational System..."):
        try:
            # Check if config.yaml exists, if not create it with defaults
            if not os.path.exists("config.yaml"):
                with open("config.yaml", "w") as f:
                    yaml.dump({
                        "knowledge_graph": {
                            "concepts_file": "data/concepts.json",
                            "questions_file": "data/questions.json",
                            "resources_file": "data/resources.json",
                            "learners_file": "data/learners.json",
                            "cache_dir": "data/cache"
                        },
                        "gnn": {
                            "model_type": "hetero_gat",
                            "hidden_channels": 64,
                            "num_layers": 2,
                            "num_heads": 4,
                            "dropout": 0.2,
                            "learning_rate": 0.001,
                            "weight_decay": 5e-4,
                            "batch_size": 32,
                            "patience": 10,
                            "validation_ratio": 0.2
                        }
                    }, f)
            
            # Create data directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
            
            # Initialize the educational system
            system = EducationalSystem("config.yaml")
            
            # Check if model exists and load it
            if os.path.exists("data/model.pt"):
                st.session_state.model_trained = system.load_model()
            
            st.session_state.system = system
            return system
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")
            return None

def get_system():
    """Get or initialize the educational system"""
    if st.session_state.system is None:
        return initialize_system()
    return st.session_state.system

def format_timestamp(timestamp_str):
    """Format a timestamp string to a readable format"""
    if not timestamp_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return timestamp_str

def download_json(data, filename):
    """Create a download link for JSON data"""
    json_str = json.dumps(data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def create_progress_bar(value, max_value=1.0, label="", color="#2196F3"):
    """Create a custom progress bar with label"""
    percentage = min(100, max(0, int(value * 100 / max_value)))
    return f"""
    <div style="margin-bottom: 10px;">
        <div style="margin-bottom: 5px;">{label}: {percentage}%</div>
        <div style="background-color: #E0E0E0; border-radius: 5px; height: 20px;">
            <div style="background-color: {color}; width: {percentage}%; height: 100%; border-radius: 5px;"></div>
        </div>
    </div>
    """

def clear_chat_history():
    """Clear the chat history"""
    st.session_state.chat_history = []
    st.session_state.current_question = None
    st.session_state.session_id = None

#--------------------------
# Sidebar Navigation
#--------------------------

def display_sidebar():
    """Display the sidebar navigation"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/240/000000/artificial-intelligence.png", width=100)
        st.title("Educational System")
        
        # System status
        if st.session_state.system is not None:
            st.success("System initialized ✓")
            
            if st.session_state.model_trained:
                st.success("Model loaded ✓")
            else:
                st.warning("Model not trained ⚠")
        else:
            if st.button("Initialize System"):
                initialize_system()
                st.rerun()
            st.error("System not initialized ✗")
            return
        
        # Navigation
        st.subheader("Navigation")
        nav_options = [
            "Dashboard",
            "Learners Management",
            "Concepts Management",
            "Interactive Learning",
            "Recommendations",
            "Model Training",
            "Data Import/Export"
        ]
        selected_nav = st.selectbox("Select a section", nav_options)
        
        # Quick actions
        st.subheader("Quick Actions")
        
        # Add example data
        if st.button("Add Example Data"):
            system = get_system()
            if system:
                with st.spinner("Adding example data..."):
                    success = system.add_example_data()
                    if success:
                        st.success("Example data added successfully!")
                    else:
                        st.error("Error adding example data.")
        
        # Train model button
        if st.button("Train Model"):
            system = get_system()
            if system:
                redirect_to_training_page()
        
        # Clear cache button
        if st.button("Clear Cache"):
            system = get_system()
            if system and hasattr(system.recommendation_system, "clear_cache"):
                system.recommendation_system.clear_cache()
                st.success("Node embedding cache cleared.")
        
        # Current selections
        if st.session_state.current_learner:
            learner = get_system().knowledge_graph.get_learner_by_id(st.session_state.current_learner)
            if learner:
                st.sidebar.markdown("---")
                st.sidebar.subheader("Current Learner")
                st.sidebar.info(f"Name: {learner.get('name', 'Unknown')}\nID: {st.session_state.current_learner}")
        
        if st.session_state.current_concept:
            concept = get_system().knowledge_graph.get_concept_by_id(st.session_state.current_concept)
            if concept:
                st.sidebar.markdown("---")
                st.sidebar.subheader("Current Concept")
                st.sidebar.info(f"Name: {concept.get('name', 'Unknown')}\nID: {st.session_state.current_concept}")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("Made with ❤️ by Streamlit and GNN")
        
        return selected_nav

def redirect_to_training_page():
    """Set navigation to model training page"""
    # This will be handled in the main app loop
    st.session_state.redirect_to = "Model Training"
    st.rerun()
#--------------------------
# Dashboard
#--------------------------

def display_dashboard():
    """Display the main dashboard with system statistics"""
    system = get_system()
    if not system:
        return
    
    st.markdown("<div class='main-header'>Educational System Dashboard</div>", unsafe_allow_html=True)
    
    # System overview
    st.markdown("<div class='section-header'>System Overview</div>", unsafe_allow_html=True)
    
    # Get counts from knowledge graph
    num_concepts = len(system.knowledge_graph.concepts.get("concepts", []))
    num_learners = len(system.knowledge_graph.learners.get("learners", []))
    num_questions = len(system.knowledge_graph.questions.get("questions", []))
    num_resources = len(system.knowledge_graph.resources.get("resources", []))
    
    # Display statistics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value'>{num_concepts}</div>
            <div class='stat-label'>Concepts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value'>{num_learners}</div>
            <div class='stat-label'>Learners</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value'>{num_questions}</div>
            <div class='stat-label'>Questions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-value'>{num_resources}</div>
            <div class='stat-label'>Resources</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display system status
    st.markdown("<div class='section-header'>System Status</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model status
        if st.session_state.model_trained:
            st.markdown("""
            <div class='success-box'>
                <h3>🧠 Model Status</h3>
                <p>The GNN model is trained and ready for recommendations.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='warning-box'>
                <h3>🧠 Model Status</h3>
                <p>The GNN model has not been trained yet. Train the model to enable recommendations.</p>
                <a href="/?nav=Model+Training">Go to Model Training</a>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Data status
        if num_concepts > 0 and num_learners > 0:
            st.markdown("""
            <div class='success-box'>
                <h3>📊 Data Status</h3>
                <p>The system has data and is ready for use.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='warning-box'>
                <h3>📊 Data Status</h3>
                <p>The system needs more data. Add concepts and learners to enable full functionality.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Recent activity
    st.markdown("<div class='section-header'>Recent Activity</div>", unsafe_allow_html=True)
    
    # Get recent learners
    learners = system.knowledge_graph.learners.get("learners", [])
    recent_learners = sorted(
        learners, 
        key=lambda l: l.get("last_activity", ""),
        reverse=True
    )[:5]
    
    if recent_learners:
        st.markdown("<h3>Recent Learner Activity</h3>", unsafe_allow_html=True)
        learner_data = []
        for learner in recent_learners:
            last_activity = format_timestamp(learner.get("last_activity", ""))
            learner_data.append({
                "ID": learner["id"],
                "Name": learner.get("name", "Unknown"),
                "Progress": f"{float(learner.get('overall_progress', 0.0))*100:.1f}%",
                "Last Activity": last_activity,
                "Points": learner.get("points", 0),
                "Streak": learner.get("study_streak", 0)
            })
        
        df = pd.DataFrame(learner_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No recent learner activity found.")
    
    # Recommendations
    st.markdown("<div class='section-header'>Quick Actions</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Manage Learners", key="dash_manage_learners"):
            st.session_state.redirect_to = "Learners Management"
            st.rerun()    
    with col2:
        if st.button("Manage Concepts", key="dash_manage_concepts"):
            st.session_state.redirect_to = "Concepts Management"
            st.rerun()    
    with col3:
        if st.button("Start Learning Session", key="dash_learning_session"):
            st.session_state.redirect_to = "Interactive Learning"
            st.rerun()
#--------------------------
# Learners Management
#--------------------------

def display_learners_management():
    """Display the learners management page"""
    system = get_system()
    if not system:
        return
    
    st.markdown("<div class='main-header'>Learners Management</div>", unsafe_allow_html=True)
    
    # Tabs for different operations
    tab1, tab2, tab3 = st.tabs(["View Learners", "Add Learner", "Learner Details"])
    
    # Tab 1: View Learners
    with tab1:
        st.markdown("<div class='section-header'>All Learners</div>", unsafe_allow_html=True)
        
        # Get all learners
        learners = system.knowledge_graph.learners.get("learners", [])
        
        if not learners:
            st.info("No learners found. Add some learners to get started.")
        else:
            # Prepare data for the table
            learner_data = []
            for learner in learners:
                last_activity = format_timestamp(learner.get("last_activity", ""))
                
                learner_data.append({
                    "ID": learner["id"],
                    "Name": learner.get("name", "Unknown"),
                    "Email": learner.get("email", ""),
                    "Progress": f"{float(learner.get('overall_progress', 0.0))*100:.1f}%",
                    "Learning Style": learner.get("learning_style", "balanced"),
                    "Points": learner.get("points", 0),
                    "Last Activity": last_activity
                })
            
            # Display as dataframe
            df = pd.DataFrame(learner_data)
            st.dataframe(df, use_container_width=True)
            
            # Learner selection
            selected_learner = st.selectbox(
                "Select a learner to view details",
                options=[l["ID"] for l in learner_data],
                format_func=lambda x: next((l["Name"] for l in learner_data if l["ID"] == x), x)
            )
            
            if st.button("View Learner Details"):
                st.session_state.current_learner = selected_learner
                st.rerun()    
    # Tab 2: Add Learner
    with tab2:
        st.markdown("<div class='section-header'>Add New Learner</div>", unsafe_allow_html=True)
        
        # Form for adding a new learner
        with st.form("add_learner_form"):
            name = st.text_input("Name", value="")
            email = st.text_input("Email", value="")
            learning_style = st.selectbox(
                "Learning Style",
                options=["visual", "balanced", "textual"],
                index=1
            )
            persistence = st.slider("Persistence", 0.0, 1.0, 0.5, 0.1)
            
            submit_button = st.form_submit_button("Add Learner")
            
            if submit_button and name:
                try:
                    learner_id = system.knowledge_graph.add_learner(
                        name=name,
                        email=email,
                        learning_style=learning_style,
                        persistence=persistence
                    )
                    st.success(f"Learner added successfully with ID: {learner_id}")
                    st.session_state.current_learner = learner_id
                except Exception as e:
                    st.error(f"Error adding learner: {str(e)}")
    
    # Tab 3: Learner Details
    with tab3:
        st.markdown("<div class='section-header'>Learner Details</div>", unsafe_allow_html=True)
        
        if not st.session_state.current_learner:
            st.info("Select a learner from the 'View Learners' tab to see details.")
        else:
            learner_id = st.session_state.current_learner
            
            # Get learner stats
            try:
                stats = system.get_learner_stats(learner_id)
                
                if "error" in stats:
                    st.error(stats["error"])
                else:
                    st.markdown(f"### {stats['name']}")
                    st.markdown(f"ID: `{learner_id}`")
                    
                    # Display basic stats in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class='stat-card'>
                            <div class='stat-value'>{stats['overall_progress']*100:.1f}%</div>
                            <div class='stat-label'>Overall Progress</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class='stat-card'>
                            <div class='stat-value'>{stats['points']}</div>
                            <div class='stat-label'>Points</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class='stat-card'>
                            <div class='stat-value'>{stats['streak']}</div>
                            <div class='stat-label'>Current Streak</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class='stat-card'>
                            <div class='stat-value'>{stats['num_mastered']}</div>
                            <div class='stat-label'>Mastered Concepts</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Mastered concepts
                    with st.expander("Mastered Concepts", expanded=False):
                        if stats['mastered_concepts']:
                            for concept in stats['mastered_concepts']:
                                st.markdown(f"""
                                <div class='concept-card'>
                                    <b>{concept['name']}</b><br>
                                    Mastery: {concept['mastery']*100:.1f}%<br>
                                    Last Updated: {format_timestamp(concept['last_updated'])}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No mastered concepts yet.")
                    
                    # In progress concepts
                    with st.expander("In Progress Concepts", expanded=False):
                        if stats['in_progress_concepts']:
                            for concept in stats['in_progress_concepts']:
                                st.markdown(f"""
                                <div class='concept-card'>
                                    <b>{concept['name']}</b><br>
                                    Mastery: {concept['mastery']*100:.1f}%<br>
                                    Last Updated: {format_timestamp(concept['last_updated'])}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No concepts in progress.")
                    
                    # Question stats
                    if 'question_stats' in stats:
                        with st.expander("Question Statistics", expanded=False):
                            q_stats = stats['question_stats']
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Questions", q_stats['total_answered'])
                            with col2:
                                st.metric("Correct Answers", q_stats['correct_answers'])
                            with col3:
                                st.metric("Accuracy", f"{q_stats['accuracy']*100:.1f}%")
                    
                    # Achievements
                    if 'achievement_stats' in stats:
                        with st.expander("Achievements", expanded=False):
                            a_stats = stats['achievement_stats']
                            st.subheader(f"Total Achievements: {a_stats['total_achievements']}")
                            
                            if 'recent_achievements' in a_stats and a_stats['recent_achievements']:
                                st.markdown("### Recent Achievements")
                                for achievement in a_stats['recent_achievements']:
                                    st.markdown(f"""
                                    <div class='achievement-card'>
                                        <b>{achievement.get('name', 'Unknown Achievement')}</b><br>
                                        Points: {achievement.get('points', 0)}<br>
                                        {achievement.get('description', '')}
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("No recent achievements.")
                    
                    # Actions
                    st.markdown("### Actions")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("Start Learning Session", key="learner_start_session"):
                            st.session_state.redirect_to = "Interactive Learning"
                            st.rerun()                    
                    with col2:
                        if st.button("Get Recommendations", key="learner_recommendations"):
                            st.session_state.redirect_to = "Recommendations"
                            st.rerun()                    
                    with col3:
                        if st.button("Generate Learning Path", key="learner_learning_path"):
                            # This will be handled in the recommendations page
                            st.session_state.redirect_to = "Recommendations"
                            st.session_state.show_learning_path = True
                            st.rerun()            
            except Exception as e:
                st.error(f"Error getting learner stats: {str(e)}")

#--------------------------
# Concepts Management
#--------------------------

def display_concepts_management():
    """Display the concepts management page"""
    system = get_system()
    if not system:
        return
    
    st.markdown("<div class='main-header'>Concepts Management</div>", unsafe_allow_html=True)
    
    # Tabs for different operations
    tab1, tab2, tab3 = st.tabs(["View Concepts", "Add Concept", "Concept Details"])
    
    # Tab 1: View Concepts
    with tab1:
        st.markdown("<div class='section-header'>All Concepts</div>", unsafe_allow_html=True)
        
        # Get all concepts
        concepts = system.knowledge_graph.concepts.get("concepts", [])
        
        if not concepts:
            st.info("No concepts found. Add some concepts to get started.")
        else:
            # Prepare data for the table
            concept_data = []
            for concept in concepts:
                prerequisites = ", ".join([p for p in concept.get("prerequisites", [])])
                
                concept_data.append({
                    "ID": concept["id"],
                    "Name": concept.get("name", "Unknown"),
                    "Difficulty": f"{float(concept.get('difficulty', 0.5)):.2f}",
                    "Complexity": f"{float(concept.get('complexity', 0.5)):.2f}",
                    "Importance": f"{float(concept.get('importance', 0.5)):.2f}",
                    "Prerequisites": prerequisites
                })
            
            # Display as dataframe
            df = pd.DataFrame(concept_data)
            st.dataframe(df, use_container_width=True)
            
            # Concept selection
            selected_concept = st.selectbox(
                "Select a concept to view details",
                options=[c["ID"] for c in concept_data],
                format_func=lambda x: next((c["Name"] for c in concept_data if c["ID"] == x), x)
            )
            
            if st.button("View Concept Details"):
                st.session_state.current_concept = selected_concept
                st.rerun()    
    # Tab 2: Add Concept
    with tab2:
        st.markdown("<div class='section-header'>Add New Concept</div>", unsafe_allow_html=True)
        
        # Form for adding a new concept
        with st.form("add_concept_form"):
            name = st.text_input("Name", value="")
            description = st.text_area("Description", value="")
            
            # Sliders for numerical values
            difficulty = st.slider("Difficulty", 0.0, 1.0, 0.5, 0.1)
            complexity = st.slider("Complexity", 0.0, 1.0, 0.5, 0.1)
            importance = st.slider("Importance", 0.0, 1.0, 0.5, 0.1)
            
            # Prerequisite selection
            all_concepts = system.knowledge_graph.concepts.get("concepts", [])
            prereq_options = {c["id"]: c.get("name", f"Concept {c['id']}") for c in all_concepts}
            
            if prereq_options:
                prerequisites = st.multiselect(
                    "Prerequisites",
                    options=list(prereq_options.keys()),
                    format_func=lambda x: prereq_options.get(x, x)
                )
            else:
                prerequisites = []
                st.info("No existing concepts to select as prerequisites.")
            
            # Option to generate content
            generate_content = st.checkbox("Automatically generate questions and resources", value=True)
            
            submit_button = st.form_submit_button("Add Concept")
            
            if submit_button and name:
                try:
                    if generate_content:
                        result = system.add_concept_with_content(
                            name=name,
                            description=description,
                            difficulty=difficulty,
                            complexity=complexity,
                            importance=importance,
                            prerequisites=prerequisites
                        )
                        concept_id = result["concept_id"]
                        st.success(f"Concept added successfully with ID: {concept_id}")
                        st.info(f"Generated {len(result.get('questions', []))} questions and {len(result.get('resources', []))} resources.")
                    else:
                        concept_id = system.knowledge_graph.add_concept(
                            name=name,
                            description=description,
                            difficulty=difficulty,
                            complexity=complexity,
                            importance=importance,
                            prerequisites=prerequisites
                        )
                        st.success(f"Concept added successfully with ID: {concept_id}")
                    
                    st.session_state.current_concept = concept_id
                    # Clear the embeddings cache
                    system.recommendation_system.clear_cache()
                except Exception as e:
                    st.error(f"Error adding concept: {str(e)}")
    
    # Tab 3: Concept Details
    with tab3:
        st.markdown("<div class='section-header'>Concept Details</div>", unsafe_allow_html=True)
        
        if not st.session_state.current_concept:
            st.info("Select a concept from the 'View Concepts' tab to see details.")
        else:
            concept_id = st.session_state.current_concept
            
            # Get concept stats
            try:
                stats = system.get_concept_stats(concept_id)
                
                if "error" in stats:
                    st.error(stats["error"])
                else:
                    st.markdown(f"### {stats['name']}")
                    st.markdown(f"ID: `{concept_id}`")
                    st.markdown(f"**Description:** {stats['description']}")
                    
                    # Display basic stats in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class='stat-card'>
                            <div class='stat-value'>{stats['difficulty']:.2f}</div>
                            <div class='stat-label'>Difficulty</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class='stat-card'>
                            <div class='stat-value'>{stats['complexity']:.2f}</div>
                            <div class='stat-label'>Complexity</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class='stat-card'>
                            <div class='stat-value'>{stats['importance']:.2f}</div>
                            <div class='stat-label'>Importance</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Prerequisites and dependencies
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Prerequisites")
                        if stats.get('prerequisites'):
                            for prereq in stats['prerequisites']:
                                st.markdown(f"""
                                <div class='concept-card'>
                                    <b>{prereq['name']}</b> ({prereq['id']})
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No prerequisites.")
                    
                    with col2:
                        st.subheader("Dependent Concepts")
                        if stats.get('dependent_concepts'):
                            for dep in stats['dependent_concepts']:
                                st.markdown(f"""
                                <div class='concept-card'>
                                    <b>{dep['name']}</b> ({dep['id']})
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No dependent concepts.")
                    
                    # Resources and questions
                    with st.expander("Related Resources", expanded=False):
                        if stats.get('related_resources'):
                            for resource in stats['related_resources']:
                                st.markdown(f"""
                                <div class='concept-card'>
                                    <b>{resource['title']}</b> (ID: {resource['id']})<br>
                                    Media Type: {resource['media_type']}<br>
                                    Quality: {resource['quality']:.2f}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No related resources.")
                    
                    with st.expander("Related Questions", expanded=False):
                        if stats.get('related_questions'):
                            for question in stats['related_questions']:
                                st.markdown(f"""
                                <div class='concept-card'>
                                    <b>{question['text']}</b> (ID: {question['id']})<br>
                                    Type: {question['type']}<br>
                                    Difficulty: {question['difficulty']:.2f}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No related questions.")
                    
                    # Learner mastery statistics
                    if 'average_mastery' in stats:
                        with st.expander("Learner Mastery Statistics", expanded=False):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Average Mastery", f"{stats['average_mastery']*100:.1f}%")
                                st.metric("Number of Learners", stats['num_learners'])
                            
                            with col2:
                                if 'mastery_distribution' in stats:
                                    dist = stats['mastery_distribution']
                                    # Create a bar chart for mastery distribution
                                    dist_data = pd.DataFrame({
                                        'Range': list(dist.keys()),
                                        'Count': list(dist.values())
                                    })
                                    
                                    fig = px.bar(
                                        dist_data, 
                                        x='Range', 
                                        y='Count',
                                        title='Mastery Distribution',
                                        labels={'Count': 'Number of Learners', 'Range': 'Mastery Range'}
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Actions
                    st.markdown("### Actions")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Generate Content", key="concept_generate_content"):
                            try:
                                content_result = system.content_generator.generate_content_for_concept(concept_id)
                                if content_result:
                                    st.success(f"Generated {len(content_result.get('questions', []))} questions and {len(content_result.get('resources', []))} resources.")
                                    # Clear the embeddings cache
                                    system.recommendation_system.clear_cache()
                                else:
                                    st.error("Error generating content.")
                            except Exception as e:
                                st.error(f"Error generating content: {str(e)}")
                    
                    with col2:
                        if st.button("Study This Concept", key="concept_study"):
                            if st.session_state.current_learner:
                                st.session_state.redirect_to = "Interactive Learning"
                                st.rerun()                            
                            else:
                                st.warning("Please select a learner first.")
            
            except Exception as e:
                st.error(f"Error getting concept stats: {str(e)}")

#--------------------------
# Interactive Learning
#--------------------------

def display_interactive_learning():
    """Display the interactive learning page with chat interface"""
    system = get_system()
    if not system:
        return
    
    st.markdown("<div class='main-header'>Interactive Learning Session</div>", unsafe_allow_html=True)
    
    # Check if we have a current learner
    if not st.session_state.current_learner:
        st.warning("Please select a learner first.")
        
        # Show learner selection
        learners = system.knowledge_graph.learners.get("learners", [])
        if learners:
            learner_options = {l["id"]: l.get("name", f"Learner {l['id']}") for l in learners}
            selected_learner = st.selectbox(
                "Select a learner",
                options=list(learner_options.keys()),
                format_func=lambda x: learner_options.get(x, x)
            )
            
            if st.button("Select Learner"):
                st.session_state.current_learner = selected_learner
                st.rerun()        
            else:
                st.error("No learners found. Please add a learner first.")
        
        return
    
    # Get the current learner
    learner = system.knowledge_graph.get_learner_by_id(st.session_state.current_learner)
    if not learner:
        st.error("Selected learner not found.")
        st.session_state.current_learner = None
        return
    
    # Display learner info
    st.markdown(f"### Learning Session for: {learner.get('name', 'Unknown')}")
    
    # Start a new session or continue existing one
    if not st.session_state.session_id:
        # Show concept selection or use recommended concept
        concepts = system.knowledge_graph.concepts.get("concepts", [])
        
        if concepts:
            # Get mastered concepts
            mastered_concepts = set()
            mastery_threshold = 0.75  # Default threshold
            
            for mastery in learner.get("concept_mastery", []):
                if float(mastery.get("level", 0)) >= mastery_threshold:
                    mastered_concepts.add(mastery["concept_id"])
            
            # Filter out mastered concepts
            available_concepts = [c for c in concepts if c["id"] not in mastered_concepts]
            
            if not available_concepts:
                st.success("Congratulations! You have mastered all available concepts.")
                return
            
            # Get recommendations if no concept selected
            if not st.session_state.current_concept:
                try:
                    recommended_concepts = system.recommendation_system.recommend_next_concepts(
                        st.session_state.current_learner, top_n=3
                    )
                    
                    if recommended_concepts:
                        st.markdown("### Recommended Concepts")
                        
                        for i, concept in enumerate(recommended_concepts):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"""
                                <div class='concept-card'>
                                    <b>{concept['name']}</b> (ID: {concept['id']})<br>
                                    Difficulty: {concept['difficulty']:.2f}, 
                                    Importance: {concept['importance']:.2f}
                                </div>
                                """, unsafe_allow_html=True)
                            with col2:
                                if st.button(f"Select", key=f"select_concept_{i}"):
                                    st.session_state.current_concept = concept['id']
                                    st.rerun()                    
                    # Allow manual selection
                    st.markdown("### Or Select a Concept Manually")
                    
                    concept_options = {c["id"]: c.get("name", f"Concept {c['id']}") for c in available_concepts}
                    selected_concept = st.selectbox(
                        "Select a concept to study",
                        options=list(concept_options.keys()),
                        format_func=lambda x: concept_options.get(x, x)
                    )
                    
                    if st.button("Start Session with Selected Concept"):
                        st.session_state.current_concept = selected_concept
                        st.rerun()                
                except Exception as e:
                    st.error(f"Error getting recommendations: {str(e)}")
                    
                    # Fallback to manual selection
                    concept_options = {c["id"]: c.get("name", f"Concept {c['id']}") for c in available_concepts}
                    selected_concept = st.selectbox(
                        "Select a concept to study",
                        options=list(concept_options.keys()),
                        format_func=lambda x: concept_options.get(x, x)
                    )
                    
                    if st.button("Start Session"):
                        st.session_state.current_concept = selected_concept
                        st.rerun()            
            # Start session if concept is selected
            elif st.session_state.current_concept:
                try:
                    session = system.interactive_session(
                        st.session_state.current_learner,
                        st.session_state.current_concept
                    )
                    
                    if "error" in session:
                        st.error(session["error"])
                    else:
                        st.session_state.session_id = session["session_id"]
                        st.session_state.current_question = session["initial_question"]
                        
                        # Add initial question to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": session["initial_question"]
                        })
                        
                        # Display learning objectives
                        if "learning_objectives" in session:
                            st.markdown("### Learning Objectives")
                            for i, objective in enumerate(session["learning_objectives"]):
                                st.markdown(f"{i+1}. {objective}")
                        
                        st.rerun()                
                except Exception as e:
                    st.error(f"Error starting session: {str(e)}")
        else:
            st.error("No concepts found. Please add some concepts first.")
    
    # Continue existing session
    else:
        # Display the chat interface
        st.markdown("### Learning Session")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Get user input
        user_input = st.chat_input("Type your answer here...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Process the response
            with st.spinner("Thinking..."):
                try:
                    result = system.process_response(
                        session_id=st.session_state.session_id,
                        learner_id=st.session_state.current_learner,
                        concept_id=st.session_state.current_concept,
                        question=st.session_state.current_question,
                        response=user_input
                    )
                    
                    # Update the current question
                    st.session_state.current_question = result["followup_question"]
                    
                    # Prepare assistant response with feedback and next question
                    response_text = ""
                    
                    # Add evaluation
                    response_text += f"**Evaluation:**\n"
                    response_text += f"* Correctness: {result['evaluation']['correctness']}/100\n"
                    response_text += f"* Reasoning: {result['evaluation']['reasoning']}/100\n"
                    response_text += f"* Overall: {result['evaluation']['overall_score']:.1f}/100\n\n"
                    
                    # Add strengths
                    response_text += f"**Strengths:**\n{result['evaluation']['strengths']}\n\n"
                    
                    # Add feedback
                    response_text += f"**Feedback:**\n{result['evaluation']['feedback']}\n\n"
                    
                    # Add achievement if earned
                    if "achievement" in result:
                        response_text += f"🏆 **Achievement Unlocked!** 🏆\n"
                        response_text += f"+{result['achievement']['points']} points: {result['achievement']['message']}\n\n"
                    
                    # Add mastery update if applicable
                    if "mastery_update" in result:
                        response_text += f"📈 **Mastery increased by {result['mastery_update']['gain']*100:.1f}%!**\n"
                        response_text += f"New mastery level: {result['mastery_update']['new_level']*100:.1f}%\n\n"
                    
                    # Add next question
                    response_text += f"**Next Question:**\n{result['followup_question']}"
                    
                    # Add assistant message to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    
                    # Display assistant message
                    with st.chat_message("assistant"):
                        st.write(response_text)
                
                except Exception as e:
                    st.error(f"Error processing response: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
        
        # Add options to end session or switch concept
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("End Session", key="end_session"):
                clear_chat_history()
                st.session_state.current_concept = None
                st.rerun()        
        with col2:
            if st.button("Switch Concept", key="switch_concept"):
                clear_chat_history()
                st.session_state.current_concept = None
                st.rerun()
#--------------------------
# Recommendations
#--------------------------

def display_recommendations():
    """Display the recommendations page"""
    system = get_system()
    if not system:
        return
    
    st.markdown("<div class='main-header'>Personalized Recommendations</div>", unsafe_allow_html=True)
    
    # Check if we have a current learner
    if not st.session_state.current_learner:
        st.warning("Please select a learner first.")
        
        # Show learner selection
        learners = system.knowledge_graph.learners.get("learners", [])
        if learners:
            learner_options = {l["id"]: l.get("name", f"Learner {l['id']}") for l in learners}
            selected_learner = st.selectbox(
                "Select a learner",
                options=list(learner_options.keys()),
                format_func=lambda x: learner_options.get(x, x)
            )
            
            if st.button("Select Learner"):
                st.session_state.current_learner = selected_learner
                st.rerun()        
            else:
                st.error("No learners found. Please add a learner first.")
        
        return
    
    # Get the current learner
    learner = system.knowledge_graph.get_learner_by_id(st.session_state.current_learner)
    if not learner:
        st.error("Selected learner not found.")
        st.session_state.current_learner = None
        return
    
    # Display learner info
    st.markdown(f"### Recommendations for: {learner.get('name', 'Unknown')}")
    
    # Check if model is trained
    if not st.session_state.model_trained:
        st.warning("The GNN model has not been trained yet. Recommendations may not be accurate.")
    
    # Tabs for different types of recommendations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Next Concepts", 
        "Resources", 
        "Learning Path",
        "Weekly Plan"
    ])
    
    # Tab 1: Next Concepts
    with tab1:
        st.markdown("<div class='section-header'>Recommended Next Concepts</div>", unsafe_allow_html=True)
        
        try:
            # Get recommended concepts
            with st.spinner("Generating recommendations..."):
                recommended_concepts = system.recommendation_system.recommend_next_concepts(
                    st.session_state.current_learner, top_n=5
                )
            
            if not recommended_concepts:
                st.info("No concept recommendations available at this time.")
            else:
                for i, concept in enumerate(recommended_concepts):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class='concept-card'>
                            <b>{concept['name']}</b> (ID: {concept['id']})<br>
                            Difficulty: {concept['difficulty']:.2f}, 
                            Importance: {concept['importance']:.2f}<br>
                            Current Mastery: {concept.get('current_mastery', 0.0)*100:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if st.button(f"Study Now", key=f"study_concept_{i}"):
                            st.session_state.current_concept = concept['id']
                            st.session_state.redirect_to = "Interactive Learning"
                            st.rerun()        
        except Exception as e:
            st.error(f"Error getting concept recommendations: {str(e)}")
    
    # Tab 2: Resources
    with tab2:
        st.markdown("<div class='section-header'>Recommended Resources</div>", unsafe_allow_html=True)
        
        # Need a concept to recommend resources
        if not st.session_state.current_concept:
            st.info("Please select a concept to get resource recommendations.")
            
            # Show concept selection
            concepts = system.knowledge_graph.concepts.get("concepts", [])
            if concepts:
                concept_options = {c["id"]: c.get("name", f"Concept {c['id']}") for c in concepts}
                selected_concept = st.selectbox(
                    "Select a concept",
                    options=list(concept_options.keys()),
                    format_func=lambda x: concept_options.get(x, x)
                )
                
                if st.button("Select Concept"):
                    st.session_state.current_concept = selected_concept
                    st.rerun()            
                else:
                    st.error("No concepts found. Please add some concepts first.")
        else:
            try:
                # Get recommended resources
                with st.spinner("Generating resource recommendations..."):
                    concept = system.knowledge_graph.get_concept_by_id(st.session_state.current_concept)
                    if not concept:
                        st.error("Selected concept not found.")
                        st.session_state.current_concept = None
                        st.rerun()                    
                    st.markdown(f"### Resources for Concept: {concept.get('name', 'Unknown')}")
                    
                    recommended_resources = system.recommendation_system.recommend_resources(
                        st.session_state.current_learner, 
                        st.session_state.current_concept,
                        top_n=5
                    )
                
                if not recommended_resources:
                    st.info("No resource recommendations available for this concept.")
                else:
                    for i, resource in enumerate(recommended_resources):
                        st.markdown(f"""
                        <div class='concept-card'>
                            <b>{resource.get('title', 'Untitled')}</b> (ID: {resource['id']})<br>
                            Media Type: {resource.get('media_type', 'unknown')}<br>
                            Quality: {resource.get('quality', 0.5):.2f}, 
                            Complexity: {resource.get('complexity', 0.5):.2f}<br>
                            <a href="{resource.get('url', '#')}" target="_blank">View Resource</a>
                        </div>
                        """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error getting resource recommendations: {str(e)}")
    
    # Tab 3: Learning Path
    with tab3:
        st.markdown("<div class='section-header'>Personalized Learning Path</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            # Target concept selection
            concepts = system.knowledge_graph.concepts.get("concepts", [])
            if concepts:
                concept_options = {c["id"]: c.get("name", f"Concept {c['id']}") for c in concepts}
                concept_options[""] = "No specific target (general path)"
                target_concept = st.selectbox(
                    "Select a target concept (optional)",
                    options=[""] + list(concept_options.keys()),
                    format_func=lambda x: concept_options.get(x, x),
                    index=0
                )
            else:
                target_concept = ""
        
        with col2:
            generate_button = st.button("Generate Learning Path")
        
        if generate_button:
            try:
                # Generate learning path
                with st.spinner("Generating learning path..."):
                    target = None if target_concept == "" else target_concept
                    path = system.generate_learning_path(st.session_state.current_learner, target)
                
                if not path:
                    st.info("No learning path could be generated. Either the target is already mastered or no suitable path exists.")
                else:
                    st.markdown("### Your Learning Path")
                    
                    for i, step in enumerate(path):
                        with st.expander(f"Step {i+1}: {step['name']}", expanded=i==0):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"**Current Mastery:** {step['current_mastery']*100:.1f}%")
                            
                            with col2:
                                st.markdown(f"**Difficulty:** {step['difficulty']:.2f}")
                            
                            with col3:
                                st.markdown(f"**Est. Study Time:** {step['estimated_study_time_minutes']} minutes")
                            
                            st.markdown(f"**Description:** {step.get('description', 'No description available.')}")
                            
                            # Recommended resources
                            if step.get('resources'):
                                st.markdown("#### Recommended Resources")
                                for res in step['resources']:
                                    st.markdown(f"* **{res['title']}** ({res['media_type']})")
                                    if 'url' in res:
                                        st.markdown(f"  [Open Resource]({res['url']})")
                            
                            # Practice questions
                            if step.get('questions'):
                                st.markdown("#### Practice Questions")
                                for q in step['questions']:
                                    st.markdown(f"* {q['text']}")
                            
                            # Study button
                            if st.button(f"Study Now", key=f"study_step_{i}"):
                                st.session_state.current_concept = step['concept_id']
                                st.session_state.redirect_to = "Interactive Learning"
                                st.rerun()            
            except Exception as e:
                st.error(f"Error generating learning path: {str(e)}")
    
    # Tab 4: Weekly Plan
    with tab4:
        st.markdown("<div class='section-header'>Weekly Study Plan</div>", unsafe_allow_html=True)
        
        if st.button("Generate Weekly Plan"):
            try:
                # Generate weekly plan
                with st.spinner("Generating weekly plan..."):
                    plan = system.generate_weekly_plan(st.session_state.current_learner)
                
                if not plan or "error" in plan:
                    st.error("Unable to generate weekly plan. Make sure there are concepts available to study.")
                else:
                    st.markdown(f"### Weekly Study Plan for {plan['learner_name']}")
                    st.markdown(f"**Period:** {plan['start_date']} to {plan['end_date']}")
                    st.markdown(f"**Total Study Time:** {plan['total_study_time_minutes']} minutes ({plan['daily_study_time_minutes']} minutes/day)")
                    
                    for day in plan['schedule']:
                        with st.expander(f"{day['day']} ({day['study_minutes']} minutes)", expanded=False):
                            for activity in day['activities']:
                                st.markdown(f"#### {activity['concept_name']} ({activity['minutes']} min)")
                                
                                # Resource
                                if activity.get('resources'):
                                    resource = activity['resources'][0]
                                    st.markdown(f"**Resource:** {resource.get('title', 'Untitled')}")
                                    if 'url' in resource:
                                        st.markdown(f"[Open Resource]({resource['url']})")
                                
                                # Question
                                if activity.get('questions'):
                                    question = activity['questions'][0]
                                    st.markdown(f"**Practice Question:** {question.get('text', '')}")
                                
                                # Study button
                                if st.button(f"Study Now", key=f"study_activity_{day['day']}_{activity['concept_id']}"):
                                    st.session_state.current_concept = activity['concept_id']
                                    st.session_state.redirect_to = "Interactive Learning"
                                    st.rerun()            
            except Exception as e:
                st.error(f"Error generating weekly plan: {str(e)}")

#--------------------------
# Model Training
#--------------------------

def display_model_training():
    """Display the model training page"""
    system = get_system()
    if not system:
        return
    
    st.markdown("<div class='main-header'>Model Training</div>", unsafe_allow_html=True)
    
    # Display model status
    if st.session_state.model_trained:
        st.markdown("""
        <div class='success-box'>
            <h3>🧠 Model Status: Trained</h3>
            <p>The GNN model is trained and ready for recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='warning-box'>
            <h3>🧠 Model Status: Not Trained</h3>
            <p>The GNN model has not been trained yet. Train the model to enable accurate recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Check data requirements
    num_concepts = len(system.knowledge_graph.concepts.get("concepts", []))
    num_learners = len(system.knowledge_graph.learners.get("learners", []))
    
    if num_concepts < 3 or num_learners < 1:
        st.markdown("""
        <div class='error-box'>
            <h3>⚠️ Insufficient Data</h3>
            <p>You need at least 3 concepts and 1 learner to train the model effectively.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Add Example Data"):
            with st.spinner("Adding example data..."):
                success = system.add_example_data()
                if success:
                    st.success("Example data added successfully!")
                else:
                    st.error("Error adding example data.")
                st.rerun()        
        return
    
    # Training options
    st.markdown("<div class='section-header'>Training Configuration</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.number_input("Number of Epochs", min_value=10, max_value=500, value=100, step=10)
    
    with col2:
        force_retrain = st.checkbox("Force Retrain", value=False, 
                                 help="Train the model even if it's already trained")
    
    # Start training
    if st.button("Start Training"):
        if st.session_state.model_trained and not force_retrain:
            st.warning("Model is already trained. Check 'Force Retrain' to train again.")
        else:
            # Train the model
            with st.spinner("Training model... This may take a while."):
                try:
                    history = system.train_model(epochs)
                    
                    if history:
                        st.session_state.model_trained = True
                        st.success(f"Model trained successfully for {len(history.get('loss', []))} epochs!")
                        
                        # Plot training history
                        if 'loss' in history:
                            history_df = pd.DataFrame({
                                'Epoch': list(range(1, len(history['loss']) + 1)),
                                'Training Loss': history['loss']
                            })
                            
                            if 'val_loss' in history:
                                history_df['Validation Loss'] = history['val_loss']
                            
                            fig = px.line(
                                history_df, 
                                x='Epoch', 
                                y=['Training Loss', 'Validation Loss'] if 'val_loss' in history else ['Training Loss'],
                                title='Training History',
                                labels={'value': 'Loss', 'variable': 'Metric'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Model training skipped (insufficient data).")
                
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
    
    # Model diagnostics
    if st.session_state.model_trained:
        st.markdown("<div class='section-header'>Model Diagnostics</div>", unsafe_allow_html=True)
        
        if st.button("Run Recommendation Diagnostics"):
            if st.session_state.current_learner:
                try:
                    with st.spinner("Running diagnostics..."):
                        diagnosis = system.diagnose_recommendations(st.session_state.current_learner)
                    
                    st.markdown("### Recommendation System Diagnosis")
                    
                    # Status
                    if diagnosis["recommendation_possible"]:
                        st.markdown("""
                        <div class='success-box'>
                            <h3>✅ Recommendation System Status: Operational</h3>
                            <p>The recommendation system is working properly.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class='error-box'>
                            <h3>❌ Recommendation System Status: Issues Detected</h3>
                            <p>There are issues with the recommendation system.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Issues
                    if diagnosis.get("issues"):
                        st.markdown("#### Issues Detected")
                        for issue in diagnosis["issues"]:
                            st.warning(issue)
                    
                    # Suggestions
                    if diagnosis.get("suggestions"):
                        st.markdown("#### Suggestions")
                        for suggestion in diagnosis["suggestions"]:
                            st.info(suggestion)
                    
                    # More details in expanders
                    if diagnosis.get("model_info"):
                        with st.expander("Model Information", expanded=False):
                            st.json(diagnosis["model_info"])
                    
                    if diagnosis.get("graph_info"):
                        with st.expander("Graph Information", expanded=False):
                            st.json(diagnosis["graph_info"])
                    
                    if diagnosis.get("content_info"):
                        with st.expander("Content Information", expanded=False):
                            st.json(diagnosis["content_info"])
                
                except Exception as e:
                    st.error(f"Error running diagnostics: {str(e)}")
            else:
                st.warning("Please select a learner first.")
        
        # Model management
        st.markdown("<div class='section-header'>Model Management</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save Model"):
                try:
                    system.save_model()
                    st.success("Model saved successfully!")
                except Exception as e:
                    st.error(f"Error saving model: {str(e)}")
        
        with col2:
            if st.button("Load Model"):
                try:
                    success = system.load_model()
                    if success:
                        st.session_state.model_trained = True
                        st.success("Model loaded successfully!")
                    else:
                        st.error("Failed to load model.")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")

#--------------------------
# Data Import/Export
#--------------------------

def display_import_export():
    """Display the data import/export page"""
    system = get_system()
    if not system:
        return
    
    st.markdown("<div class='main-header'>Data Import & Export</div>", unsafe_allow_html=True)
    
    # Tabs for import and export
    tab1, tab2 = st.tabs(["Import", "Export"])
    
    # Tab 1: Import
    with tab1:
        st.markdown("<div class='section-header'>Import Data</div>", unsafe_allow_html=True)
        
        # Import concepts
        with st.expander("Import Concepts", expanded=True):
            st.markdown("""
            Upload a JSON file containing concepts to import.
            
            The file should have this structure:
            ```json
            {
                "concepts": [
                    {
                        "id": "concept1",
                        "name": "Concept Name",
                        "description": "Description of the concept",
                        "difficulty": 0.5,
                        "complexity": 0.5,
                        "importance": 0.5,
                        "prerequisites": ["prereq1", "prereq2"]
                    }
                ]
            }
            ```
            """)
            
            uploaded_file = st.file_uploader("Choose a concepts JSON file", type="json")
            
            col1, col2 = st.columns(2)
            
            with col1:
                overwrite = st.checkbox("Overwrite existing concepts", value=False)
            
            with col2:
                generate_content = st.checkbox("Generate content for imported concepts", value=True)
            
            if uploaded_file is not None:
                if st.button("Import Concepts"):
                    try:
                        # Save uploaded file to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
                            f.write(uploaded_file.getvalue())
                            temp_path = f.name
                        
                        # Import concepts
                        with st.spinner("Importing concepts..."):
                            added, skipped = system.import_concepts_from_file(temp_path, overwrite)
                            
                            st.success(f"Import complete: {added} concepts added, {skipped} skipped.")
                            
                            # Generate content if requested
                            if generate_content and added > 0:
                                st.info("Generating content for imported concepts...")
                                
                                try:
                                    # Load the imported concepts
                                    with open(temp_path, 'r') as f:
                                        import_data = json.load(f)
                                    
                                    concepts = import_data.get("concepts", [])
                                    generated_count = 0
                                    
                                    progress_bar = st.progress(0)
                                    
                                    for i, concept in enumerate(concepts):
                                        concept_id = concept["id"]
                                        if concept_id in system.knowledge_graph._concept_lookup:
                                            try:
                                                result = system.content_generator.generate_content_for_concept(concept_id)
                                                if result:
                                                    generated_count += 1
                                            except Exception as e:
                                                st.warning(f"Error generating content for {concept.get('name', concept_id)}: {str(e)}")
                                        
                                        # Update progress
                                        progress_bar.progress((i + 1) / len(concepts))
                                    
                                    st.success(f"Content generation complete for {generated_count} concepts.")
                                    
                                    # Clear the embeddings cache
                                    system.recommendation_system.clear_cache()
                                    
                                except Exception as e:
                                    st.error(f"Error generating content: {str(e)}")
                            
                            # Remove the temporary file
                            os.unlink(temp_path)
                    
                    except Exception as e:
                        st.error(f"Error importing concepts: {str(e)}")
        
        # Example data button
        with st.expander("Add Example Data", expanded=False):
            st.markdown("""
            Add example data to the system for testing purposes.
            
            This will add:
            - Several concepts with prerequisites
            - Questions for each concept
            - Learning resources
            - Example learners with progress
            """)
            
            if st.button("Add Example Data", key="import_example_data"):
                with st.spinner("Adding example data..."):
                    success = system.add_example_data()
                    if success:
                        st.success("Example data added successfully!")
                    else:
                        st.error("Error adding example data.")
    
    # Tab 2: Export
    with tab2:
        st.markdown("<div class='section-header'>Export Data</div>", unsafe_allow_html=True)
        
        # Export all data
        with st.expander("Export All Data", expanded=True):
            if st.button("Export All Data"):
                try:
                    # Get all data
                    export_data = {
                        "concepts": system.knowledge_graph.concepts.get("concepts", []),
                        "questions": system.knowledge_graph.questions.get("questions", []),
                        "resources": system.knowledge_graph.resources.get("resources", []),
                        "learners": system.knowledge_graph.learners.get("learners", [])
                    }
                    
                    # Create download links
                    st.markdown(download_json(export_data, "educational_system_data.json"), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error exporting data: {str(e)}")
        
        # Export specific data
        with st.expander("Export Specific Data", expanded=False):
            data_type = st.selectbox(
                "Select data type to export",
                options=["concepts", "questions", "resources", "learners"]
            )
            
            if st.button("Export Selected Data"):
                try:
                    # Get selected data
                    if data_type == "concepts":
                        export_data = system.knowledge_graph.concepts
                    elif data_type == "questions":
                        export_data = system.knowledge_graph.questions
                    elif data_type == "resources":
                        export_data = system.knowledge_graph.resources
                    elif data_type == "learners":
                        export_data = system.knowledge_graph.learners
                    
                    # Create download link
                    st.markdown(download_json(export_data, f"{data_type}.json"), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error exporting data: {str(e)}")

#--------------------------
# Main Application
#--------------------------

def main():
    """Main application function"""
    # Set up the sidebar navigation
    selected_nav = display_sidebar()
    
    # Check for redirects
    if hasattr(st.session_state, 'redirect_to'):
        selected_nav = st.session_state.redirect_to
        del st.session_state.redirect_to
    
    # Display the selected section
    if selected_nav == "Dashboard":
        display_dashboard()
    elif selected_nav == "Learners Management":
        display_learners_management()
    elif selected_nav == "Concepts Management":
        display_concepts_management()
    elif selected_nav == "Interactive Learning":
        display_interactive_learning()
    elif selected_nav == "Recommendations":
        display_recommendations()
    elif selected_nav == "Model Training":
        display_model_training()
    elif selected_nav == "Data Import/Export":
        display_import_export()

if __name__ == "__main__":
    main()