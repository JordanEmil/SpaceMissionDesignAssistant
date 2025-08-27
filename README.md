 # Space Mission Design Assistant

A Retrieval-Augmented Generation (RAG) chatbot for querying historical space mission data. This system provides intelligent answers about satellite missions, orbits, payloads, and mission designs by leveraging a comprehensive knowledge base of space missions.

## Features

- **Interactive Chatbot Interface**: Streamlit-based web interface for easy interaction
- **Comprehensive Knowledge Base**: Contains data from hundreds of space missions
- **RAG Architecture**: Combines vector search with LLM generation for accurate, contextual responses
- **Source Attribution**: All answers include references to source missions
- **Optimised Performance**: Fine-tuned retrieval parameters for best results

## Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SpaceMissionDesignAssistant
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Streamlit Chatbot

1. Ensure you have the ChromaDB index ready (the `chroma_db` directory should exist)

2. Launch the Streamlit application:
```bash
streamlit run streamlit_chatbot.py
```

3. Open your browser and navigate to `http://localhost:8501`

4. Enter your OpenAI API key when prompted

5. Start asking questions about space missions!

### Example Questions

- "What orbit regimes have been used for SAR imaging satellites?"
- "What are typical power requirements for Earth observation CubeSats?"
- "Which missions have used optical imaging payloads?"
- "What are common failure modes in small satellite missions?"
- "Compare antenna designs used in different SAR missions"

## Project Structure

```
SpaceMissionDesignAssistant/
├── streamlit_chatbot.py      # Main Streamlit application
├── query_pipeline.py          # Core RAG query engine
├── evaluation_framework.py    # Evaluation metrics and framework
├── chroma_db/                 # Vector database storage
├── chat_logs/                 # Conversation history logs
├── data/                      # Raw mission data
├── rag_ready_data/           # Processed documents for indexing
├── Indexing/                  # Indexing scripts
│   ├── indexing_pipeline.py   # Main indexing script
│   └── prepare_rag_data.py    # Data preparation utilities
├── Scraping/                  # Web scraping utilities
└── evaluation/                # Evaluation suite and results
```

## Configuration

The chatbot uses optimised parameters determined through extensive evaluation:
- **Top-K Retrieval**: 5 documents
- **Similarity Threshold**: 0.5
- **Temperature**: 0.1
- **LLM Model**: OpenAI GPT (configurable)

## Building the Knowledge Base

If you need to rebuild the index:

1. Prepare the documents:
```bash
python Indexing/prepare_rag_data.py
```

2. Build the ChromaDB index:
```bash
python Indexing/indexing_pipeline.py
```

## Advanced Usage

### Evaluation Framework

The project includes a comprehensive evaluation framework for testing different parameter configurations:

```bash
cd evaluation
python run_evaluation_pipeline.py
```

### Custom Queries

You can also use the query pipeline programmatically:

```python
from query_pipeline import SpaceMissionQueryEngine

# Initialise engine
engine = SpaceMissionQueryEngine(
    chroma_persist_dir="./chroma_db",
    top_k=5,
    similarity_threshold=0.5,
    temperature=0.1
)

# Query
result = engine.query("What are the main components of a SAR satellite?")
print(result['response'])
```

## Requirements

Main dependencies:
- streamlit
- llama-index
- chromadb
- openai
- pandas
- numpy

See `requirements.txt` for a complete list.

## Data Sources

The knowledge base is built from publicly available space mission data, including mission specifications, technical details, and historical information from various space agencies and organisations.

## License

This project is provided as-is for educational and research purposes.

## Author

Emil Ares

## Troubleshooting

### Common Issues

1. **"ChromaDB directory not found"**: Run the indexing pipeline first to create the database
2. **"Invalid API key"**: Ensure your OpenAI API key is valid and has sufficient credits
3. **"Module not found"**: Install all requirements using `pip install -r requirements.txt`

### Performance Tips

- The first query may take longer as the system loads the index
- For faster responses, consider using a local LLM or adjusting the `top_k` parameter
- Chat history is automatically saved in the `chat_logs` directory