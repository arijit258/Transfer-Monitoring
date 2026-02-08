# Transformer Predictive Maintenance System

An AI-powered web application for real-time monitoring, health assessment, and predictive maintenance of power transformers. Built with Django and integrated with **Ollama** (local) and **Groq** (cloud) LLMs through a Retrieval-Augmented Generation (RAG) pipeline.

## Screenshots

### Dashboard
Real-time fleet overview with status distribution charts, health score visualizations, and transformer cards showing live health percentages.

![Dashboard](Screenshot/Screenshot%202026-02-08%20204457.png)

### Transformer Fleet List
Filterable table of all monitored transformers with status indicators, health scores, scenario tags, and color-coded severity levels.

![Transformer Fleet](Screenshot/Screenshot%202026-02-08%20204511.png)

### AI Chat Assistant
Conversational AI assistant powered by RAG with support for multiple LLM providers (Ollama local models and Groq cloud models), sample questions, and transformer-specific context.

![AI Chat Assistant](Screenshot/Screenshot%202026-02-08%20204532.png)

### Transformer Detail View
Individual transformer profile with specifications, live sensor readings, risk factors, maintenance recommendations, assessment summaries, and DGA analysis charts.

![Transformer Detail](Screenshot/Screenshot%202026-02-08%20204558.png)

## Features

- **Real-Time Dashboard** - Fleet-wide monitoring with status distribution (Healthy/Warning/Critical), average health scores, and interactive Chart.js visualizations
- **Transformer Detail Pages** - Individual profiles with sensor readings, DGA gas levels, oil quality metrics, risk factors, and maintenance recommendations
- **AI Chat Assistant (RAG)** - Ask natural language questions about transformer health, DGA interpretation, maintenance planning, and fault diagnosis with context retrieved from transformer-specific documents
- **Dual LLM Support** - Choose between local Ollama models (phi3, mistral, llama3) or Groq cloud models (Llama 3.3 70B, Llama 3.1 8B, Mixtral 8x7B, Gemma 7B)
- **Vector Search with ChromaDB** - Document embeddings via HuggingFace Sentence Transformers with ChromaDB for semantic retrieval over DGA reports, maintenance logs, and PDF assessments
- **7 Realistic Scenarios** - Pre-built transformer scenarios covering normal operation, insulation degradation, core ground fault, moisture ingress, partial discharge, standby condition, and LTC degradation
- **REST API** - JSON endpoints for transformer data, sensor readings, dashboard stats, chart data, and chat interactions
- **Chat History Persistence** - Conversations stored in the database with session management

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Django 4.2+ / Python |
| Database | SQLite (default) |
| AI/RAG | LangChain, ChromaDB, HuggingFace Sentence Transformers |
| LLM Providers | Ollama (local), Groq (cloud) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Frontend | Django Templates, Tailwind CSS, Chart.js |
| Data Formats | CSV, PDF, Markdown, JSON, TXT |

## Project Structure

```
Transformers Predictive Maintainance/
├── manage.py
├── requirements.txt
├── db.sqlite3
├── transformer_pm/               # Django project settings
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── monitoring/                   # Main Django app
│   ├── models.py                 # Transformer, SensorReading, RiskFactor, etc.
│   ├── views.py                  # Dashboard, detail, chat, and API views
│   ├── urls.py                   # URL routing
│   ├── admin.py
│   ├── ai/
│   │   └── rag_engine.py         # Agentic RAG engine, Ollama/Groq clients
│   ├── templates/monitoring/
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── transformer_list.html
│   │   ├── transformer_detail.html
│   │   ├── chat.html
│   │   └── transformer_chat.html
│   ├── management/commands/
│   │   ├── load_scenarios.py     # Load transformer data from JSON
│   │   └── load_transformers.py
│   └── migrations/
├── data_source/
│   ├── scenarios/
│   │   └── transformers_scenarios.json   # 7 transformer scenarios
│   ├── transformers/             # Per-transformer data files
│   │   ├── TR-001/               # DGA history, sensor readings, maintenance logs
│   │   ├── TR-002/
│   │   ├── TR-003/
│   │   ├── TR-004/
│   │   ├── TR-005/
│   │   ├── TR-006/
│   │   └── TR-007/
│   ├── reports/                  # Markdown assessment reports
│   └── pdfs/                     # PDF assessment documents
├── chroma_db/                    # ChromaDB vector store (per transformer)
└── Screenshot/                   # Application screenshots
```

## Installation

### Prerequisites

- Python 3.10+
- (Optional) [Ollama](https://ollama.com) for local LLM inference
- (Optional) [Groq API Key](https://console.groq.com/) for cloud LLM inference

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Transformers Predictive Maintainance"
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
   pip install -r requirements.txt
   ```

3. **Configure environment variables** (optional, for Groq cloud models)
   ```bash
   # Create a .env file in the project root
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run database migrations**
   ```bash
   python manage.py migrate
   ```

5. **Load transformer scenario data**
   ```bash
   python manage.py load_scenarios
   ```

6. **Start the development server**
   ```bash
   python manage.py runserver
   ```

7. **Open the application** at `http://127.0.0.1:8000/`

### (Optional) Setup Ollama for Local Models

```bash
# Install and start Ollama
ollama serve

# Pull a model
ollama pull phi3
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard |
| `/transformers/` | GET | Transformer fleet list |
| `/transformer/<id>/` | GET | Transformer detail page |
| `/chat/` | GET | Global AI chat interface |
| `/chat/<id>/` | GET | Transformer-specific chat |
| `/api/transformers/` | GET | All transformers (JSON) |
| `/api/transformers/<id>/readings/` | GET | Sensor readings (JSON) |
| `/api/stats/` | GET | Dashboard statistics (JSON) |
| `/api/chat/` | POST | Send question to AI |
| `/api/chat/history/clear/` | POST | Clear chat session |
| `/api/models/` | GET | Available AI models |
| `/api/chart-data/<id>/` | GET | DGA/sensor chart data |
| `/api/sample-questions/` | GET | Sample questions by category |

## Monitored Transformers

| ID | Name | Rating | Status | Scenario |
|----|------|--------|--------|----------|
| TR-001 | Alpha Substation Main | 50 MVA | Healthy | Normal Operation |
| TR-002 | Beta Industrial Transformer | 75 MVA | Warning | Insulation Degradation |
| TR-003 | Gamma Distribution Unit | 100 MVA | Critical | Core Ground Fault |
| TR-004 | Delta Regional Transformer | 40 MVA | Warning | Moisture Ingress |
| TR-005 | Epsilon Power Station Unit | 150 MVA | Critical | Partial Discharge Activity |
| TR-006 | Zeta Backup Unit | 30 MVA | Healthy | Standby - Excellent Condition |
| TR-007 | Eta Transmission Autotransformer | 200 MVA | Warning | LTC Degradation + Oil Contamination |

## AI Capabilities

The RAG-powered AI assistant can answer questions across five categories:

- **Conceptual** - Transformer health factors, load impact, protection systems
- **Data-Based** - DGA trend interpretation, oil quality correlation, health index computation
- **Scenario** - Fault diagnosis from DGA patterns, BDV decline analysis, moisture ingress causes
- **Forecasting** - Health index prediction, maintenance scheduling, degradation rate estimation
- **Diagnostic** - Fault type identification (thermal/electrical/arcing), partial discharge detection

## License

This project is for educational and research purposes.
