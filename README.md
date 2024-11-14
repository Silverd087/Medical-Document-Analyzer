# Medical Document Analyzer

A Streamlit application that analyzes medical documents using local LLMs through Ollama.

## Features

- Document Analysis (PDF/TXT)
- Direct Text Input
- Real-time Processing Status
- Incremental Results Display
- PDF/Text Report Generation

## Requirements

- Python 3.8+
- Ollama
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/medical-document-analyzer.git
   cd medical-document-analyzer
   ```

2. Create and activate virtual environment:

   ```bash
   python -m venv venv

   # Linux/Mac:
   source venv/bin/activate

   # Windows:
   venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install Ollama from [ollama.ai](https://ollama.ai)

5. Pull required models:
   ```bash
   ollama pull llama2
   ollama pull mistral
   ollama pull neural-chat
   ollama pull openchat
   ```

## Usage

Run the Streamlit app:

```bash
streamlit run interface.py
```

## Project Structure

medical-document-analyzer/
├── app.py # Core processing logic
├── interface.py # Streamlit interface
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── .gitignore # Git ignore rules

## License

MIT License
