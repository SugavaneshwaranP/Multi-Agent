# CognifyX - Multi-Agent Automation System

CognifyX is a multi-agent automation system for Sales Analytics & Resume Screening using CrewAI, AutoGen, and local LLMs via Ollama.

## Installation

1. Create Virtual Environment
```
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Install Ollama

### Windows
Download from https://ollama.ai/download and install.

### macOS
```
brew install ollama
```

### Linux
```
curl -fsSL https://ollama.ai/install.sh | sh
```

4. Pull models
```
ollama pull llama3
ollama pull mistral
ollama pull qwen2.5
```

## Usage

Run sales analytics:
```
python main.py --task sales
```

Run resume screening:
```
python main.py --task resume
