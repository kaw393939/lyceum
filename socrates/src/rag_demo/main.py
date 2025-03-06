import os
import asyncio
import streamlit as st
import yaml
import uuid
from dotenv import load_dotenv

# Import our custom services
from rag_demo.utils import RAGService
from rag_demo.services.chat_service import ChatService
from rag_demo.models.chat import ChatHistory

# Load environment variables from .env file
load_dotenv()

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    # Initialize user ID for persistence
    if "user_id" not in st.session_state:
        # In a real app, this would come from authentication
        st.session_state.user_id = f"anon_{uuid.uuid4().hex[:8]}"
    
    # Initialize active chat ID
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None
    
    # Initialize chat service
    if "chat_service" not in st.session_state:
        use_mock = os.environ.get("MONGODB_USE_MOCK", "true").lower() in ("true", "1")
        st.session_state.chat_service = ChatService(use_mock=use_mock)
    
    # Initialize transient chat history (will be replaced by persistent one)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize RAG service
    if "rag_service" not in st.session_state:
        try:
            # Attempt to create RAG service
            # Store flag to prevent coroutine warning when setting rag_initialized to False
            rag_success = True
            st.session_state.rag_service = RAGService()
            
            # Register session state callback for when session ends
            def on_shutdown():
                if "rag_service" in st.session_state:
                    # Use run_coroutine_threadsafe to properly cleanup async tasks
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            st.session_state.rag_service.cleanup(),
                            loop
                        )
            
            # Register the callback function to execute when Streamlit script reruns
            st.session_state["_on_shutdown"] = on_shutdown
            st.session_state.rag_initialized = True
        except Exception as e:
            st.error(f"Failed to initialize RAG service: {e}")
            st.session_state.rag_initialized = False
    
    # Initialize sources tracking
    if "current_sources" not in st.session_state:
        st.session_state.current_sources = []
        
    # Initialize request stats tracking
    if "last_request_stats" not in st.session_state:
        st.session_state.last_request_stats = {}
    
    # Initialize RAG toggle
    if "use_rag" not in st.session_state:
        st.session_state.use_rag = True
        
    # Initialize chat list
    if "chat_list" not in st.session_state:
        st.session_state.chat_list = []

def load_configuration():
    """Load configuration from config.yaml with error handling."""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.warning("Config file not found. Using default settings.")
        return {
            "system_prompt": "You are a helpful assistant.",
            "model": "gpt-4o",
            "temperature": 0.7
        }
    except yaml.YAMLError as e:
        st.error(f"Error parsing config file: {e}")
        return {
            "system_prompt": "You are a helpful assistant.",
            "model": "gpt-4o",
            "temperature": 0.7
        }

async def generate_response(user_message, config):
    """Generate a response using the RAG service."""
    try:
        # First check if health check needs to be started (only do this once)
        if hasattr(st.session_state.rag_service, 'start_health_check'):
            await st.session_state.rag_service.start_health_check()
        
        # Convert chat history to the format expected by the RAG service
        formatted_history = []
        for msg in st.session_state.chat_history:
            formatted_history.append(msg)
        
        # Generate response using RAG
        response, sources, request_stats = await st.session_state.rag_service.generate_completion(
            user_message=user_message,
            system_prompt=config.get("system_prompt", "You are a helpful assistant."),
            chat_history=formatted_history,
            retrieve=st.session_state.use_rag
        )
        
        # Save sources and request stats for display
        st.session_state.current_sources = sources
        st.session_state.last_request_stats = request_stats
        
        return response
        
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return f"I'm sorry, I encountered an error: {str(e)}"

def send_message(user_message):
    """Handle a new message submission and get a response using RAG."""
    # Validate input
    if not user_message:
        return

    # Check if RAG service is initialized
    if not st.session_state.rag_initialized:
        st.error("RAG service is not properly initialized. Check the API key and connections.")
        return
    
    # Make sure we have an active chat
    if not st.session_state.active_chat_id:
        asyncio.run(create_new_chat())

    # Load configuration
    config = load_configuration()

    # Display user message immediately
    st.chat_message("user").markdown(user_message)
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_message})
    
    # Persist the user message
    asyncio.run(st.session_state.chat_service.add_message(
        st.session_state.active_chat_id,
        "user",
        user_message
    ))
    
    # Show generating indicator
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Generating response...")
        
        # Run async generation
        response = asyncio.run(generate_response(user_message, config))
        
        # Display response
        message_placeholder.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Persist the assistant response
    asyncio.run(st.session_state.chat_service.add_message(
        st.session_state.active_chat_id,
        "assistant",
        response
    ))
    
    # Update chat list to show latest chats
    asyncio.run(load_user_chats())

async def load_or_create_chat():
    """Load the active chat or create a new one if none exists."""
    if st.session_state.active_chat_id:
        # Try to load the active chat
        chat = await st.session_state.chat_service.get_chat(st.session_state.active_chat_id)
        if chat:
            st.session_state.chat_history = [msg.to_llm_format() for msg in chat.messages]
            return
    
    # If no active chat or couldn't load, create a new one
    await create_new_chat()

async def create_new_chat():
    """Create a new chat and set it as active."""
    # Create a new chat
    chat = await st.session_state.chat_service.create_chat(st.session_state.user_id)
    st.session_state.active_chat_id = chat.id
    st.session_state.chat_history = []
    st.session_state.current_sources = []

async def load_user_chats():
    """Load the list of chats for the current user."""
    chats = await st.session_state.chat_service.get_user_chats(st.session_state.user_id)
    st.session_state.chat_list = chats

async def switch_chat(chat_id):
    """Switch to a different chat."""
    # Save any messages in the current chat before switching
    if st.session_state.active_chat_id and st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            await st.session_state.chat_service.add_message(
                st.session_state.active_chat_id,
                msg["role"],
                msg["content"]
            )
    
    # Set the new active chat
    st.session_state.active_chat_id = chat_id
    await st.session_state.chat_service.set_active_chat(st.session_state.user_id, chat_id)
    
    # Load the new chat
    chat = await st.session_state.chat_service.get_chat(chat_id)
    if chat:
        st.session_state.chat_history = [msg.to_llm_format() for msg in chat.messages]
    else:
        st.session_state.chat_history = []
    
    # Clear sources
    st.session_state.current_sources = []
    
    # Rerun to update UI
    st.rerun()

async def delete_current_chat():
    """Delete the current chat and switch to a new one."""
    if st.session_state.active_chat_id:
        await st.session_state.chat_service.delete_chat(st.session_state.active_chat_id)
    
    # Create a new chat
    await create_new_chat()
    await load_user_chats()
    
    # Rerun to update UI
    st.rerun()

def clear_chat():
    """Clear the chat history for the current chat."""
    st.session_state.chat_history = []
    st.session_state.current_sources = []
    
    # Run async function to create a new chat
    asyncio.run(create_new_chat())

def main():
    """Main function for the Streamlit chat application."""
    # Set page configuration
    st.set_page_config(
        page_title="Goliath Educational AI", 
        page_icon="🧠",
        layout="wide"
    )

    # Initialize session state
    initialize_session_state()
    
    # Initialize persistent chat history
    asyncio.run(load_or_create_chat())
    
    # Load user's chat list
    asyncio.run(load_user_chats())

    # Page title and description
    st.title("🧠 Goliath Educational Platform")
    st.markdown("Chat with an AI enhanced with educational content from the Goliath knowledge base.")

    # Create three columns - sidebar, chat main area, and sources sidebar
    chat_area, sources_area = st.columns([3, 1])
    
    # Create the chat area
    with chat_area:
        # API Key check
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OpenAI API key is missing. Please set it in your .env file.")
            return
    
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.chat_message("user").markdown(message['content'])
            else:
                st.chat_message("assistant").markdown(message['content'])
    
        # User input
        user_input = st.chat_input("Ask a question about Stoicism or educational topics...")
        if user_input:
            send_message(user_input)
    
    # Create the sidebar
    with st.sidebar:
        st.title("Chat History")
        
        # New chat button
        if st.button("🆕 New Chat", use_container_width=True):
            asyncio.run(create_new_chat())
            asyncio.run(load_user_chats())
            st.rerun()
            
        # Display list of previous chats
        if st.session_state.chat_list:
            st.markdown("### Previous Chats")
            
            for chat in st.session_state.chat_list:
                # Highlight the active chat
                is_active = chat.id == st.session_state.active_chat_id
                
                # Create a container for each chat with conditional styling
                chat_container = st.container(border=is_active)
                with chat_container:
                    col1, col2 = st.columns([4, 1])
                    
                    # Chat title and timestamp
                    with col1:
                        # Format the timestamp
                        timestamp = chat.updated_at.strftime("%m/%d %H:%M")
                        
                        # Show preview of last message if available
                        preview = ""
                        if chat.messages:
                            last_msg = chat.messages[0]
                            preview = last_msg.content[:30] + "..." if len(last_msg.content) > 30 else last_msg.content
                        
                        # Create a clickable title
                        if st.button(f"**{chat.title}**\n{timestamp}\n{preview}", key=f"chat_{chat.id}", use_container_width=True):
                            asyncio.run(switch_chat(chat.id))
                    
                    # Delete button 
                    with col2:
                        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
                        if st.button("🗑️", key=f"delete_{chat.id}"):
                            if chat.id == st.session_state.active_chat_id:
                                asyncio.run(delete_current_chat())
                            else:
                                asyncio.run(st.session_state.chat_service.delete_chat(chat.id))
                                asyncio.run(load_user_chats())
                                st.rerun()
        
        st.divider()
        
        st.title("Settings")
        
        # RAG toggle
        st.checkbox("Use knowledge retrieval", value=True, key="use_rag")
        
        # Navigation/utility buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear Chat"):
                clear_chat()
        with col2:
            if st.button("🔍 Diagnostics"):
                # Using query params for navigation in Streamlit
                st.query_params["page"] = "diagnostics"
                st.rerun()
    
    # Create the sources sidebar
    with sources_area:
        # Show request stats if available
        if st.session_state.last_request_stats:
            with st.expander("Request Statistics"):
                stats = st.session_state.last_request_stats
                st.markdown(f"**Total Time:** {stats.get('total_time', 0):.2f}s")
                if "retrieval_stats" in stats:
                    st.markdown(f"**Sources Used:** {stats['retrieval_stats'].get('sources_count', 0)}")
                    st.markdown(f"**Retrieval Time:** {stats['retrieval_stats'].get('retrieval_time', 0):.2f}s")
                if "llm_stats" in stats:
                    st.markdown("**LLM Usage:**")
                    st.markdown(f"- Prompt Tokens: {stats['llm_stats'].get('prompt_tokens', 0)}")
                    st.markdown(f"- Completion Tokens: {stats['llm_stats'].get('completion_tokens', 0)}")
                    st.markdown(f"- Total Tokens: {stats['llm_stats'].get('total_tokens', 0)}")
        
        # Show sources if available
        if st.session_state.current_sources:
            st.markdown("### Sources Used")
            
            for i, source in enumerate(st.session_state.current_sources):
                with st.expander(f"{i+1}. {source['metadata'].get('title', 'Source')}"):
                    st.markdown(f"**Source:** {source['metadata'].get('source', 'Unknown')}")
                    st.markdown(f"**Type:** {source['metadata'].get('type', 'Unknown')}")
                    st.markdown(f"**Relevance:** {source.get('score', 0):.2f}")
                    st.markdown("**Content:**")
                    st.markdown(source['content'])

if __name__ == "__main__":
    main()