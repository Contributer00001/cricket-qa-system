
# Cricket Statistics Q&A System 🏏

Fine-tuned LLM system for answering cricket match statistics questions using a multi-agent architecture with grounded answer extraction.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/🤗-Model-yellow.svg)](https://huggingface.co/Conqueror00001/qwen-ipl-lora)

## 🎯 Features

- **Multi-Agent Architecture**: Deterministic stats computation + LLM reasoning
- **Fine-tuned Model**: LoRA-adapted Qwen 0.5B optimized for cricket statistics
- **Grounded Answers**: Strict extraction prevents hallucinations
- **Lazy Loading**: Fast startup, model loads on-demand
- **Docker Ready**: One-command deployment
- **Sample Data**: Works out-of-box for testing

## 🏗️ Architecture
```
User Question
    ↓
┌───────────────────────────────────┐
│ StatsTool                         │
│ (Deterministic computation)       │
│ - Runs in last N overs            │
│ - Total runs/wickets              │
│ - Fours, sixes, etc.              │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│ RetrieverAgent                    │
│ (Context generation)              │
│ - Formats stats as text           │
│ - Creates LLM-readable context    │
└───────────────────────────────────┘
    ↓
┌───────────────────────────────────┐
│ AnalystAgent                      │
│ (LLM + Grounded extraction)       │
│ - Fine-tuned Qwen 0.5B            │
│ - LoRA adapters                   │
│ - Regex-based grounding           │
└───────────────────────────────────┘
    ↓
Numeric Answer (grounded in context)
```

## 🚀 Quick Start

### Prerequisites

- Docker (recommended)
- OR Python 3.10+ with pip
- HuggingFace account ([create here](https://huggingface.co/join))

### Option 1: Docker (Recommended)
```bash
# 1. Get HuggingFace token
# Visit: https://huggingface.co/settings/tokens
# Create token with "Read" permission
export HF_TOKEN="hf_your_token_here"

# 2. Build image
docker build -t cricket-qa:latest -f service/Dockerfile .

# 3. Run container
docker run -d \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  --name cricket-qa \
  cricket-qa:latest

# 4. Test (first request takes 2-5 min to load model)
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"question": "How many runs in the last 5 overs?"}'
```

### Option 2: Local Development
```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/cricket-qa-system.git
cd cricket-qa-system

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
export HF_TOKEN="hf_your_token_here"

# 5. Run service
uvicorn service.app:app --host 0.0.0.0 --port 8000

# 6. Test in another terminal
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"question": "How many wickets were taken?"}'
```

## 📊 Sample Data

Included sample data (`data/sample_match.json`):
- **30 cricket events** across 5 overs
- **Total runs**: 47
- **Wickets**: 3
- **Fours**: 4
- **Sixes**: 3

Perfect for testing without real match data.

## 🔧 API Endpoints

### `GET /healthz`
Health check (always responds immediately)
```bash
curl http://localhost:8000/healthz
```
Response: `{"status": "ok"}`

### `GET /readyz`
Readiness check (returns true after model loads)
```bash
curl http://localhost:8000/readyz
```
Response: `{"ready": true, "loading": false}`

### `POST /infer`
Main inference endpoint

**Request:**
```json
{
  "question": "How many runs in the last 5 overs?",
  "max_tokens": 100,
  "commentary": {
    "commentaries": [
      {"event": "ball", "over": 1, "run": 4, "four": true, "six": false},
      ...
    ]
  }
}
```

**Response:**
```json
{
  "answer": "47",
  "context_used": "Context:\n- Total runs: 47\n- Runs in last 5 overs: 47...",
  "status": "success"
}
```

**Note**: `commentary` is optional. If not provided, uses default sample data.

## 📁 Project Structure
```
cricket-qa-system/
├── agents/
│   ├── __init__.py
│   └── multi_agent.py          # StatsTool, RetrieverAgent, AnalystAgent
├── service/
│   ├── app.py                  # FastAPI service (lazy loading)
│   └── Dockerfile              # Docker configuration
├── api/
│   └── main.py                 # Alternative API entrypoint
├── scripts/
│   ├── train_sft.py            # LoRA training script
│   ├── prepare_dataset.py      # Data preparation
│   └── generate_qa.py          # QA pair generation
├── tests/
│   └── test_inference.py       # Pytest tests
├── data/
│   └── sample_match.json       # Demo data (included)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## 🤖 Model Details

- **Base Model**: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) via PEFT
- **Adapter Repo**: [Conqueror00001/qwen-ipl-lora](https://huggingface.co/Conqueror00001/qwen-ipl-lora)
- **Training**: Supervised fine-tuning on cricket Q&A pairs
- **Grounding**: Strict numeric extraction (only returns numbers present in context)
- **Device**: CPU/MPS (Apple Silicon) compatible

## 🔒 Using Your Own Data

### Data Format

Your cricket commentary data should follow this JSON structure:
```json
{
  "commentaries": [
    {
      "event": "ball",
      "over": 1,
      "run": 4,
      "four": true,
      "six": false
    },
    {
      "event": "wicket",
      "over": 2,
      "run": 0,
      "four": false,
      "six": false
    }
  ]
}
```

### Option 1: Environment Variable
```bash
export DEFAULT_MATCH_FILE="/path/to/your/match.json"
uvicorn service.app:app --host 0.0.0.0 --port 8000
```

### Option 2: Docker Volume Mount
```bash
docker run -d \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e DEFAULT_MATCH_FILE="/data/your_match.json" \
  -v /path/on/host:/data \
  cricket-qa:latest
```

### Option 3: Request Body

Include commentary directly in API request:
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many runs?",
    "commentary": {
      "commentaries": [...]
    }
  }'
```

## 🧪 Testing
```bash
# Run tests
pytest tests/

# Specific test
pytest tests/test_inference.py -v

# With coverage
pytest --cov=agents --cov=service tests/
```

## 🎓 Training Your Own Model

If you want to train on your own cricket data:
```bash
# 1. Prepare dataset
python scripts/prepare_dataset.py

# 2. Generate Q&A pairs
python scripts/generate_qa.py

# 3. Train LoRA adapters
python scripts/train_sft.py

# 4. Upload to HuggingFace (optional)
python scripts/upload_to_hf.py
```

**Note**: Training data not included in this repo. You'll need your own cricket commentary data.

## 🐛 Troubleshooting

### Model Loading Fails
```
Error: Can't find 'adapter_config.json'
```
**Solution**: Verify HF_TOKEN is set and has read access:
```bash
echo $HF_TOKEN
hf auth whoami
```

### Port Already in Use
```
Error: Address already in use
```
**Solution**: Use different port:
```bash
docker run -p 8001:8000 ...  # Use 8001 instead
```

### First Request Timeout
First request takes 2-5 minutes to download model (~500MB). Subsequent requests are instant (<1s).

### Wrong Answers
The model is grounded to only return numbers from context. If context doesn't contain the answer, it returns "The information is not available in the provided data."

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Base Model**: Qwen by Alibaba Cloud
- **Fine-tuning**: PEFT library by HuggingFace
- **Framework**: FastAPI by Tiangolo

## 🔗 Links

- [HuggingFace Model](https://huggingface.co/Conqueror00001/qwen-ipl-lora)
- [Base Model (Qwen)](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [PEFT Library](https://github.com/huggingface/peft)

## 📧 Questions?

Open an issue on GitHub or reach out via HuggingFace discussions.

---
---

## Authors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Contributer00001">
        <img src="https://github.com/Contributer00001.png" width="100px;" alt="Parth Giri"/>
        <br />
        <sub><b>Parth Giri</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/namankhatakdotcpp">
        <img src="https://github.com/namankhatakdotcpp.png" width="100px;" alt="Naman Khatak"/>
        <br />
        <sub><b>Naman Khatak</b></sub>
      </a>
    </td>
  </tr>
</table>

**Made with ❤️ by Parth Giri and Naman Khatak**
**Built with ❤️ for cricket analytics**
