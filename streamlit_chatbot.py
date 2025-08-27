#!/usr/bin/env python3
"""
Space Mission RAG Chatbot - Streamlit Interface
Web-based interface for querying space mission knowledge base
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import hashlib

import streamlit as st
from query_pipeline import SpaceMissionQueryEngine
from llama_index.core.response_synthesizers import ResponseMode


class StreamlitSpaceMissionChatbot:
    """Streamlit-based chatbot for space mission queries"""
    
    def __init__(
        self,
        chroma_persist_dir: str = "./chroma_db",
        log_dir: str = "./chat_logs",
        auto_save: bool = True
    ):
        """
        Initialize the Streamlit chatbot interface
        
        Args:
            chroma_persist_dir: Directory where ChromaDB is persisted
            log_dir: Directory to store chat logs
            auto_save: Whether to automatically save conversations
        """
        self.chroma_persist_dir = chroma_persist_dir
        self.log_dir = Path(log_dir)
        self.auto_save = auto_save
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.messages = []
            st.session_state.session_id = self._generate_session_id()
            st.session_state.session_start = datetime.now()
            st.session_state.show_sources = True
            st.session_state.query_count = 0
            st.session_state.api_key = None
            st.session_state.query_engine = None
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]
    
    def _save_conversation(self, force: bool = False):
        """Save the conversation to a JSON file"""
        if not st.session_state.messages:
            return
            
        if not self.auto_save and not force:
            return
            
        # Prepare conversation data
        conversation_data = {
            'session_id': st.session_state.session_id,
            'session_start': st.session_state.session_start.isoformat(),
            'session_end': datetime.now().isoformat(),
            'num_queries': st.session_state.query_count,
            'messages': st.session_state.messages
        }
        
        # Generate filename
        timestamp = st.session_state.session_start.strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"chat_log_{timestamp}_{st.session_state.session_id}.json"
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(conversation_data, f, indent=2)
            
        return filename
    
    def _format_sources(self, sources: List[Dict]) -> str:
        """Format source documents for display with deduplication"""
        if not sources:
            return ""
            
        # Deduplicate sources based on mission_id (since same mission can have multiple chunks)
        seen_missions = {}
        unique_sources = []
        
        for source in sources:
            metadata = source.get('metadata', {})
            mission_id = metadata.get('mission_id', '')
            
            # Use mission_id as the primary deduplication key
            if mission_id:
                if mission_id not in seen_missions:
                    seen_missions[mission_id] = source
                    unique_sources.append(source)
                elif source.get('score', 0) > seen_missions[mission_id].get('score', 0):
                    # Replace with higher scoring chunk from same mission
                    idx = unique_sources.index(seen_missions[mission_id])
                    unique_sources[idx] = source
                    seen_missions[mission_id] = source
            else:
                # If no mission_id, add it anyway (shouldn't happen in practice)
                unique_sources.append(source)
            # Limit to 20 unique sources
            if len(unique_sources) >= 20:
                break
        
        # Sort by score descending
        unique_sources.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        formatted = "\n\n**Sources:**"
        for i, source in enumerate(unique_sources, 1):
            metadata = source.get('metadata', {})
            title = metadata.get('title', 'Unknown Mission')
            url = metadata.get('url', '')
            score = source.get('score', 0)
            
            # Clean up title if needed
            if title and ' - eoPortal' in title:
                title = title.replace(' - eoPortal', '')
            
            # Format with URL if available
            if url:
                # Ensure URL is properly formatted
                if not url.startswith('http'):
                    url = f"https://{url}"
                formatted += f"\n{i}. [{title}]({url}) (relevance: {score:.3f})"
            else:
                formatted += f"\n{i}. **{title}** (relevance: {score:.3f})"
        
        return formatted
    
    def _process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return the result"""
        try:
            result = st.session_state.query_engine.query(
                query,
                response_mode=ResponseMode.COMPACT,
                return_sources=True,
                verbose=False
            )
            
            # Increment query count
            st.session_state.query_count += 1
            
            # Auto-save conversation if enabled
            if self.auto_save:
                self._save_conversation()
                
            return result
            
        except Exception as e:
            return {
                'response': f"Error processing query: {str(e)}",
                'sources': [],
                'metadata': {'response_time': 0}
            }
    
    
    def render_sidebar(self):
        """Render the sidebar with settings and information"""
        with st.sidebar:
            st.header("Space Mission Chatbot")
            
            # Session info
            st.subheader("Session Info")
            st.text(f"Session ID: {st.session_state.session_id}")
            st.text(f"Queries: {st.session_state.query_count}")
            
            # Settings
            st.subheader("Settings")
            st.session_state.show_sources = st.checkbox(
                "Show source documents",
                value=st.session_state.show_sources
            )
            
            
            # Actions
            st.subheader("Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Clear Chat"):
                    st.session_state.messages = []
                    st.session_state.query_count = 0
                    st.rerun()
            
            with col2:
                if st.button("Save Chat"):
                    filename = self._save_conversation(force=True)
                    if filename:
                        st.success(f"Saved to {filename.name}")
            
            # Examples
            st.subheader("Example Questions")
            examples = [
                "What orbit regimes have been used for SAR imaging satellites?",
                "What are typical power requirements for Earth observation CubeSats?",
                "Which missions have used optical imaging payloads?",
                "What are common failure modes in small satellite missions?",
                "Compare antenna designs used in different SAR missions"
            ]
            
            for example in examples:
                if st.button(example, key=f"example_{hash(example)}"):
                    st.session_state.example_query = example
                    st.rerun()
            
            # Stats
            if st.checkbox("Show Engine Stats"):
                stats = st.session_state.query_engine.get_engine_stats()
                st.json(stats)
    
    def render_api_key_input(self):
        """Render the API key input interface"""
        st.title("Space Mission Design Assistant - Authored by Emil Ares")
        st.markdown("Welcome! Please enter your OpenAI API key to begin.")
        
        with st.form("api_key_form"):
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Your API key will be stored only for this session"
            )
            
            st.markdown("""
            **Note:** 
            - Your API key is needed to process queries using OpenAI's language models
            - The key is stored only in your current session and is not saved permanently
            - You can get an API key from [OpenAI's platform](https://platform.openai.com/api-keys)
            """)
            
            submitted = st.form_submit_button("Initialize Chatbot")
            
            if submitted:
                if api_key and api_key.startswith("sk-"):
                    st.session_state.api_key = api_key
                    st.rerun()
                else:
                    st.error("Please enter a valid OpenAI API key (should start with 'sk-')")
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show metadata if available
                if "metadata" in message and message["role"] == "assistant":
                    st.caption(f"Response time: {message['metadata']['response_time']:.2f}s")
                    
                # Display formatted sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    # Sources are already included in the content with proper formatting
                    pass
        
        # Handle example query if set
        if hasattr(st.session_state, 'example_query'):
            query = st.session_state.example_query
            del st.session_state.example_query
        else:
            # Chat input
            query = st.chat_input(
                "Ask about space missions, orbits, payloads, etc..."
            )
        
        if query:
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(query)
            
            # Process query with loading indicator
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    result = self._process_query(query)
                
                # Display response
                response_text = result['response']
                
                # Add sources if enabled
                if st.session_state.show_sources and result.get('sources'):
                    response_text += self._format_sources(result['sources'])
                
                st.write(response_text)
                
                # Show metadata
                query_time = result['metadata'].get('response_time', 0)
                st.caption(f"Response time: {query_time:.2f}s")
                
                # Add assistant message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": result['metadata'],
                    "sources": result.get('sources', [])
                })
    
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="Space Mission Chatbot - Authored by Emil Ares",
            page_icon="🚀",
            layout="wide"
        )
        
        # Apply custom CSS for better styling
        st.markdown("""
        <style>
        .stChatMessage {
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Check for API key first
        if not st.session_state.api_key:
            self.render_api_key_input()
            return
        
        # Initialize query engine after API key is provided
        if not st.session_state.initialized and st.session_state.api_key:
            try:
                # Set the API key in environment
                os.environ["OPENAI_API_KEY"] = st.session_state.api_key
                
                with st.spinner("Initializing Space Mission Assistant..."):
                    st.session_state.query_engine = SpaceMissionQueryEngine(
                        chroma_persist_dir=self.chroma_persist_dir,
                        top_k=5, # Retrieve top 5 documents for better context OPTIMAL RESULT 
                        similarity_threshold=0.5, # OPTIMAL RESULT
                        temperature=0.1, # OPTIMAL RESULT
                        llm_model="o3"  # Use o3 for user-facing chatbot as more powerful model, 
                        # note we used chatgpt 4o-mini for our evaluations in the IRP due to cost constraints
                    )
                    st.session_state.initialized = True
                    st.success("Successfully initialized!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error initializing query engine: {e}")
                st.session_state.api_key = None
                st.stop()
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content area
        st.title("Space Mission Design Assistant - Authored by Emil Ares")
        st.markdown("Ask questions about historical space missions, orbits, payloads, and mission designs.")
        
        # Render chat interface
        self.render_chat_interface()


def main():
    """Main function to run the Streamlit chatbot"""
    # Check if index exists
    chroma_persist_dir = "./chroma_db"
    if not Path(chroma_persist_dir).exists():
        st.error(f"Error: ChromaDB directory {chroma_persist_dir} not found")
        st.info("Please run Indexing/indexing_pipeline.py first to create the index")
        st.stop()
    
    # Create and run chatbot
    chatbot = StreamlitSpaceMissionChatbot(chroma_persist_dir=chroma_persist_dir)
    chatbot.run()


if __name__ == "__main__":
    main()