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

```plaintext
medical-document-analyzer/
├── app.py                # Core processing logic
├── interface.py          # Streamlit interface
├── requirements.txt      # Project dependencies
├── README.md            # Project documentation
└── .gitignore           # Git ignore rules
```

## License

MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
