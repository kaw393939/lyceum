# rag_demo

A Python project

## Development Setup

1. Clone the repository
2. Create and activate virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install development dependencies:
   ```
   pip install -r requirements-dev.txt
   pip install -e .
   ```

## Running the Streamlit Application

Run the Streamlit application:
```
./run_streamlit.sh
```

Or directly with the streamlit command:
```
streamlit run app.py
```

## Running Tests

Run tests with pytest:
```
pytest
```

Or with coverage:
```
pytest --cov=rag_demo
```

## Docker

Build and run with Docker:
```
docker build -t rag_demo .
docker run rag_demo
```


## Project Summary

This project was created with the Python Project Generator script.
