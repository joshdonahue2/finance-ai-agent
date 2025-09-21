# Finance AI Agent: Self-Hosted Bank Statement Analyzer

> **💰 Process scanned bank statements → Extract transactions → Generate AI-powered financial insights → Beautiful dashboard**  
> *Fully dockerized, no cloud dependencies, works with your existing Ollama + Supabase*

## 🎯 What It Does

This project transforms your scanned PDF bank statements into actionable financial insights:

### Features
- **📁 PDF Upload**: Manual upload via beautiful web form (supports your 2 banks)
- **🔍 OCR Processing**: Handles scanned PDFs with Tesseract OCR
- **💳 Transaction Extraction**: Smart regex parsing tuned for bank formats
- **🧠 AI Insights**: RAG-powered analysis with Ollama (categorization, trends, anomalies, budgeting, tax deductions)
- **📊 Beautiful Visualizations**: Plotly charts (spending pies, monthly trends, anomaly detection)
- **🗄️ Vector Storage**: Embeds transactions in your Supabase pgvector database
- **⚡ Batch Processing**: Load 7+ years of historical data in one go

### Supported Insights
- **Categorization**: Groceries, Dining, Utilities, etc. (LLM-powered)
- **Trends**: Monthly spending patterns with line charts
- **Anomalies**: Statistical outliers (>2σ) and large transactions
- **Budgeting**: Personalized recommendations based on your spending
- **Tax Deductions**: Flags potential write-offs (home office, donations, etc.)

## 🏗️ Architecture

```
[Browser] → [Streamlit Dashboard:8501] 
           ├── Upload PDF → [Processor: OCR → Extract → Embed] 
           │                ↓
           ├── [Your Ollama:11434] ← Embeddings (nomic-embed-text) + LLM (llama3)
           └── [Your Supabase] ← Vector DB (pgvector) + REST API
```

**Data Flow**: Upload → OCR (Tesseract) → Extract (Pandas/Regex) → Chunk & Embed (Ollama) → Store (Supabase) → Query/RAG → Visualize (Plotly)

## 🚀 Quick Start

### Prerequisites
- [x] **Ollama running** with models: `nomic-embed-text` (embeddings) + `llama3` (LLM)
- [x] **Supabase running** with pgvector extension enabled
- [ ] **Docker** installed on your machine

### 1. Clone & Setup
```bash
# Create project directory
mkdir finance-ai-agent && cd finance-ai-agent

# Create folders
mkdir -p src uploads initial_pdfs supabase
```

### 2. Configure Environment
Create `.env` with your **existing service details**:
```bash
# Your self-hosted Supabase (REST API, not direct PostgreSQL)
SUPABASE_URL=https://supabase.donahuenet.xyz
SUPABASE_KEY=your_service_role_key_or_anon_key

# Your existing Ollama
OLLAMA_HOST=http://host.docker.internal:11434

# OCR settings
TESSERACT_LANG=eng
```

**Get your Supabase key**: Settings → API → `service_role` key (recommended for write access)

### 3. Save Project Files
Copy these files into your project:

**`requirements.txt`**:
```txt
streamlit==1.28.0
pandas==2.0.3
pdf2image==1.16.3
pytesseract==0.3.10
pillow==10.0.0
supabase==1.0.3
plotly==5.15.0
numpy==1.24.3
psycopg2-binary==2.9.7
ollama==0.1.4
python-dotenv==1.0.0
```

**`Dockerfile`**:
```dockerfile
FROM python:3.10-slim

# Install system dependencies for OCR and PDF processing
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create upload directories
RUN mkdir -p /app/uploads /app/initial_pdfs

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
CMD ["streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

**`docker-compose.yml`**:
```yaml
version: '3.8'
services:
  finance-app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./uploads:/app/uploads
      - ./initial_pdfs:/app/initial_pdfs
    env_file:
      - .env
    restart: unless-stopped
    # Optional: Add network if your services use custom Docker networks
    # networks:
    #   - supabase-net

  # Optional: Separate batch processor for initial data load
  batch-processor:
    build: .
    volumes:
      - ./initial_pdfs:/app/initial_pdfs
    env_file:
      - .env
    command: ["python", "src/process_pdf.py", "--batch", "/app/initial_pdfs"]
    profiles:
      - batch
```

**Database Schema** (`supabase/init.sql` - run once in Supabase SQL editor):
```sql
-- Enable pgvector if not already done
CREATE EXTENSION IF NOT EXISTS vector;

-- Create transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id BIGSERIAL PRIMARY KEY,
    pdf_filename TEXT NOT NULL,
    bank_name TEXT,
    transaction_date DATE,
    description TEXT,
    amount DECIMAL(10,2),
    category TEXT,
    full_chunk TEXT,
    embedding VECTOR(384),  -- nomic-embed-text dimension
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create index for fast similarity search
CREATE INDEX IF NOT EXISTS transactions_embedding_idx 
ON transactions USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Create indexes for queries
CREATE INDEX IF NOT EXISTS idx_transaction_date ON transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_bank_name ON transactions(bank_name);
CREATE INDEX IF NOT EXISTS idx_category ON transactions(category);
```

### 4. Build & Run
```bash
# Build and start dashboard
docker-compose up --build -d

# Verify it's running
curl http://localhost:8501/healthz
```

### 5. Access Dashboard
- **URL**: http://localhost:8501
- **Upload**: Use sidebar to process individual PDFs
- **Batch**: Use "Batch Processing" tab for multiple files

## 📦 Initial Data Load (7 Years of Statements)

For your ~84 PDFs covering 7 years:

### Option 1: Dashboard Batch Upload
1. Go to **Batch Processing** tab
2. Upload multiple PDFs at once
3. Click "Process X files" (processes ~10-20 at a time to avoid timeouts)

### Option 2: Command Line Batch
```bash
# 1. Put all PDFs in ./initial_pdfs/
# 2. Run batch processor
docker-compose run --rm finance-app python src/process_pdf.py --batch ./initial_pdfs

# Monitor progress
docker-compose logs -f finance-app
```

**Expected Time**: ~2-3 hours total (OCR is the bottleneck, ~1-2min per PDF)

## 🔧 Customization

### Tuning for Your Bank PDFs

1. **Test OCR on Sample**:
```bash
# Create a test file with your sample PDF
docker-compose run --rm finance-app python -c "
from process_pdf import PDFProcessor
p = PDFProcessor()
text = p.ocr_pdf('/app/initial_pdfs/your_sample.pdf')
print('=== OCR OUTPUT ===')
print(text[:2000])
"
```

2. **Update Bank Patterns** (`src/process_pdf.py`):
```python
# For Bank A - update regex pattern
def extract_transactions_bank_a(self, text: str) -> pd.DataFrame:
    # Your specific pattern: e.g., r'(\d{2}-\d{2}-\d{4})\s+(.+?)\s+\$(\d+\.\d{2})'
    pattern = r'YOUR_CUSTOM_PATTERN_HERE'
    # ... rest of extraction logic
```

3. **Bank Detection**:
```python
def detect_bank_from_filename(self, filename: str) -> str:
    filename_lower = filename.lower()
    if 'chase' in filename_lower or 'your_bank_a' in filename_lower:
        return 'Bank A'
    elif 'wells' in filename_lower or 'your_bank_b' in filename_lower:
        return 'Bank B'
    return 'Bank A'  # Default
```

### Development Commands

```bash
# Watch logs
docker-compose logs -f finance-app

# Enter container for debugging
docker-compose exec finance-app bash

# Test single PDF
docker-compose run --rm finance-app python src/process_pdf.py --pdf ./sample.pdf

# Test Supabase connection
docker-compose run --rm finance-app python -c "
from supabase import create_client
import os
client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
print('✅ Supabase connected!')
print(f'Current transactions: {client.table(\"transactions\").select(\"count\").execute().count}')
"

# Clear cache (if dashboard seems stale)
docker-compose restart finance-app
```

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| **"Connection failed"** | Check `.env` has correct `SUPABASE_URL` (HTTPS) and `SUPABASE_KEY` |
| **"No transactions extracted"** | Run OCR test above, tune regex patterns in `process_pdf.py` |
| **"Embedding failed"** | Verify Ollama models: `curl http://localhost:11434/api/tags` |
| **"Tesseract not found"** | Rebuild container: `docker-compose up --build` |
| **Slow processing** | OCR is CPU-intensive; process in smaller batches |
| **Dashboard blank** | Check logs: `docker-compose logs finance-app` |

### Common Fixes

**Supabase Connection**:
```bash
# Test API access
curl -H "apikey: $SUPABASE_KEY" \
     "https://supabase.donahuenet.xyz/rest/v1/?select=*" 
```

**Ollama Health**:
```bash
# Check models
curl http://localhost:11434/api/tags

# Pull if missing
curl -X POST http://localhost:11434/api/pull -d '{"name": "nomic-embed-text"}'
curl -X POST http://localhost:11434/api/pull -d '{"name": "llama3"}'
```

## 📱 Dashboard Walkthrough

Once running, you'll see:

### **Sidebar Controls**
- **📁 File Upload**: Drop PDFs + select bank
- **🚀 Process**: Triggers OCR → extraction → embedding
- **⚙️ Settings**: Auto-categorize, refresh data

### **Main Dashboard Tabs**

1. **📈 Overview**
   - Spending pie chart by category
   - Monthly trends line chart
   - Bank comparison bar chart
   - Key metrics (total spent, avg monthly)

2. **🔍 Insights**
   - **🤖 AI Chat**: Ask "Show my biggest expenses" or "Budget recommendations"
   - **Quick buttons**: Spending summary, budget tips, tax deductions
   - Source transactions expandable

3. **📋 Anomalies**
   - Statistical outliers (>2σ from mean)
   - Large transaction review (adjustable threshold)
   - Interactive scatter plot

4. **⚙️ Batch Processing**
   - Upload multiple PDFs at once
   - Process local folder (for initial data load)
   - Progress tracking

## 🔒 Security & Privacy

- **✅ All local**: No data leaves your machine
- **✅ Self-hosted**: Supabase + Ollama run on your infrastructure
- **✅ No cloud**: Docker containers, no external APIs
- **🔐 API keys**: Store in `.env`, not committed to git
- **🗑️ Temp files**: PDFs deleted after processing

**To add basic auth** (optional):
```python
# Add to dashboard.py top
import streamlit_authenticator as stauth

# In .env: USERNAME=your_username PASSWORD=your_password
# Then wrap dashboard content with: if stauth.authenticate():
```

## 📈 Performance Notes

- **Single PDF**: ~1-2 minutes (mostly OCR)
- **84 PDFs batch**: ~2-3 hours total (process in batches of 10-20)
- **Query response**: <5 seconds (RAG retrieval + LLM)
- **Storage**: ~100KB per transaction (text + 384-dim vector)
- **Memory**: ~2GB RAM for processing, 1GB for dashboard

## 🤝 Contributing

1. **Tune extraction**: Update regex patterns for your specific bank formats
2. **Add categories**: Extend LLM prompt for domain-specific tags
3. **Custom insights**: Modify `generate_insights()` prompt for your needs
4. **Export features**: Add CSV/PDF export buttons
5. **Advanced viz**: Sankey diagrams, forecast charts

## 📄 License
MIT - Feel free to adapt for personal use or contribute back!

## 🙏 Acknowledgments
Built with ❤️ using:
- [Streamlit](https://streamlit.io) - Beautiful web dashboard
- [Ollama](https://ollama.ai) - Local LLM + embeddings
- [Supabase](https://supabase.com) - Self-hosted vector database
- [Tesseract](https://github.com/tesseract-ocr) - OCR for scanned PDFs
- [Plotly](https://plotly.com/python) - Stunning interactive charts

---

**💡 Pro Tip**: Start with 1-2 sample PDFs to tune the extraction patterns, then batch-process your full 7-year history. The AI insights get smarter with more data!

**🚀 Ready to analyze your finances?** `docker-compose up` and upload your first statement!