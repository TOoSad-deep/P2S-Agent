# P2S-Agent: PNG to Shader Pipeline Agent

A standalone pipeline agent that converts PNG images to GLSL shaders using a deterministic candidate pool with optional LLM stages, orchestrated by LangGraph.

## Features

- **LangGraph Pipeline**: Core pipeline orchestrated by LangGraph StateGraph
- **Multiple Candidates**: Baseline, rule-based, CV, LLM, and fallback candidates
- **Quality Routing**: Automatic quality assessment and optimization
- **DSL Compiler**: Deterministic DSL-to-GLSL compilation
- **Web UI**: React-based frontend with real-time preview

## Architecture

```
Input PNG → Preprocess → Candidates → Scoring → Selection → [Optimization/Revision/Refinement]
                ↓
        LangGraph StateGraph
```

### Pipeline Stages

1. **Preprocess**: Extract features from input image (colors, edges, alpha coverage)
2. **Candidates**: Generate multiple shader candidates using different strategies
3. **Scoring**: Evaluate candidates with objective metrics (MSE, SSIM, etc.)
4. **Selection**: Select the best candidate based on quality routing
5. **Optimization**: Coordinate descent on DSL parameters (optional)
6. **Revision**: Structured DSL patches for quality improvement (optional)
7. **Refinement**: LLM-driven closed-loop refinement (optional)

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### Installation

```bash
# Clone the repository
cd P2S-Agent

# Install all dependencies
./start.sh install

# Or install separately
cd backend && pip install -r requirements.txt
cd frontend && npm install
```

### Configuration

Copy the example environment file and configure your API keys:

```bash
cp backend/.env.example backend/.env
# Edit backend/.env with your API keys
```

### Running

```bash
# Start both backend and frontend
./start.sh start

# Or start separately
./start.sh backend   # Backend on port 8001
./start.sh frontend   # Frontend on port 5174
```

### API Endpoints

- `POST /png-shader/run` - Submit image for processing
- `GET /png-shader/status/{run_id}` - Check pipeline status
- `POST /png-shader/refine/{run_id}` - Trigger LLM refinement
- `GET /api/strategy-config` - Get strategy configuration

## Project Structure

```
P2S-Agent/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Configuration
│   │   ├── state.py             # LangGraph state definition
│   │   ├── pipeline/            # Pipeline orchestration
│   │   │   ├── graph.py         # LangGraph StateGraph
│   │   │   ├── pool.py          # Candidate pool management
│   │   │   ├── scoring.py       # Evaluation and scoring
│   │   │   └── ...
│   │   ├── candidates/          # Candidate generators
│   │   ├── dsl/                 # DSL schema and compiler
│   │   ├── metrics/             # Quality metrics
│   │   └── services/            # External services
│   ├── tests/                   # Unit tests
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Main application
│   │   ├── components/          # React components
│   │   ├── hooks/               # Custom hooks
│   │   └── lib/                 # Utilities
│   ├── package.json
│   └── vite.config.ts
└── start.sh                     # Startup script
```

## Testing

```bash
cd backend
python -m pytest tests/unit/ -v
```

## License

Internal use only.
