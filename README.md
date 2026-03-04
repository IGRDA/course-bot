# Course Generator Agent

An AI-powered course generation system that creates comprehensive educational content from topics or PDF syllabi. Built with LangGraph for orchestration and supporting multiple LLM providers.

## ✨ Features

- **Multiple Input Sources**: Generate courses from topics or PDF syllabi
- **Complete Course Pipeline**: Research → Index → Theory → Activities → HTML → Images → Bibliography
- **Multi-Provider LLM Support**: Mistral, Gemini, OpenAI, Groq, DeepSeek
- **Podcast Generation**: Two-speaker dialogue with Edge TTS or Coqui TTS audio synthesis
- **PDF Book Export**: LaTeX-based professional PDF book generation
- **Bibliography Generation**: Automatic book recommendations via Open Library API with APA 7 citations
- **Vision-Powered Image Selection**: Optional Pixtral-based image ranking
- **Evaluation Framework**: LLM-as-judge quality assessment with rubric scoring
- **LangSmith Tracing**: Full observability for debugging and monitoring

## 📦 Installation

### 1. Clone and Install (Lightweight — Text Generation Only)

The default install includes everything needed for text-based course generation (LangChain, LLM providers, web search, TTS, image search). No heavy ML packages.

```bash
git clone <repository-url>
cd course-generator-agent
pip3 install -e .
```

### 2. Install Browser Automation (for image search)

```bash
playwright install chromium
```

### 3. Optional Dependency Groups

Heavy packages are split into optional groups to keep the default install lightweight:

```bash
# PDF to Markdown extraction (docling + easyocr) — adds ~2 GB
pip3 install -e ".[pdf-extraction]"

# ML / Transformers (torch + torchaudio + transformers) — adds ~2 GB
pip3 install -e ".[ml]"

# Full install: all heavy optional deps
pip3 install -e ".[all]"

# macOS native OCR (Apple Vision framework, macOS only)
pip3 install -e ".[macos-ocr]"

# Evaluation framework
pip3 install -e ".[evaluation]"

# Coqui TTS (local TTS, conflicts with docling — use separate venv)
pip3 install -e ".[coqui-tts]"
```

**Summary of dependency groups:**

| Group | Packages | Size | Use Case |
|-------|----------|------|----------|
| *(default)* | LangChain, pydantic, playwright, edge-tts, etc. | ~500 MB | Text course generation, podcasts |
| `pdf-extraction` | docling, easyocr | ~2 GB | PDF syllabus → Markdown conversion |
| `ml` | torch, torchaudio, transformers | ~2 GB | ML inference, Coqui TTS backend |
| `all` | pdf-extraction + ml | ~4 GB | Full local development |
| `macos-ocr` | ocrmac | ~50 MB | macOS-only native OCR |
| `evaluation` | textstat, langdetect, nltk, sentence-transformers | ~500 MB | Course quality evaluation |
| `coqui-tts` | TTS | ~1 GB | Local TTS (conflicts with docling) |

### 4. System Dependencies

**PDF Book Generation** requires LaTeX:

```bash
# macOS
brew install --cask basictex

# Ubuntu/Debian
sudo apt-get install texlive-latex-base texlive-fonts-recommended texlive-latex-extra biber

# Windows
# Install MiKTeX from https://miktex.org/download
```

### 5. Docker (Lightweight Image)

For containerized deployments that only need text generation (no PDF extraction or ML):

```dockerfile
RUN git clone <repository-url> /app/course-agent && \
    pip install --no-cache-dir -e /app/course-agent
```

This installs only the default lightweight dependencies. To include heavy deps:

```dockerfile
# With PDF extraction support
pip install --no-cache-dir -e "/app/course-agent[pdf-extraction]"

# Full install
pip install --no-cache-dir -e "/app/course-agent[all]"
```

## 🔐 Environment Setup

1. Create your environment file:

```bash
cp env.example env.secrets
```

2. Add your API keys to `env.secrets`:

```bash
# LLM Providers (add keys for providers you'll use)
export MISTRAL_API_KEY=your_key
export GEMINI_API_KEY=your_key
export OPENAI_API_KEY=your_key
export GROQ_API_KEY=your_key
export DEEPSEEK_API_KEY=your_key

# Vision LLM (for image ranking)
export PIXTRAL_API_KEY=your_key

# Web Search (optional, DDG works without key)
export TAVILY_API_KEY=your_key

# LangSmith Tracing (optional)
export LANGCHAIN_TRACING_V2=true
export LANGSMITH_API_KEY=your_key
```

3. Load environment:

```bash
source env.secrets
```

## 🚀 Usage

### Generate Course from Topic

Full pipeline: Research → Index → Theory → Activities → HTML → Images → Bibliography

```bash
python3 -m main.workflow --total-pages 10
```

### Generate Course from PDF Syllabus

Extract structure from PDF and generate content:

```bash
python3 -m main.workflow_pdf
```

### Generate Podcast from Topic

Streamlined pipeline for podcast output (skips activities/HTML/images):

```bash
python3 -m main.workflow_podcast --total-pages 5 --target-words 600 --tts-engine edge
```

### Generate Podcast from PDF

PDF syllabus to podcast conversion:

```bash
python3 -m main.workflow_pdf2podcast example_pdfs/coaching_y_orientacion.pdf \
    --total-pages 50 \
    --target-words 600 \
    --tts-engine edge \
    --language "Español"
```

## 🛠️ Post-Processing Tools

### Generate PDF Book

Convert completed course to professional PDF book:

```bash
# Basic usage
python3 -m tools.json2book output/MyCourseName/course.json

# With options
python3 -m tools.json2book output/MyCourseName/course.json \
    --template academic \
    --output-dir ./my_book \
    --no-images
```

### Generate Podcast for Module

Create podcast audio from existing course:

```bash
# Generate podcast for module 1
python3 -m agents.podcast_generator output/MyCourseName/course.json --module 1

# With options
python3 -m agents.podcast_generator output/MyCourseName/course.json \
    --module 2 \
    --provider openai \
    --target-words 800 \
    --skip-tts  # Only generate conversation JSON
```

### Simplify Module for Video

Extract essential fields for video generation:

```bash
# Save simplified JSON
python3 -m agents.video_html_generator output/MyCourseName/module_0.json

# Preview without saving
python3 -m agents.video_html_generator output/MyCourseName/module_0.json --preview
```

## 📊 Evaluation Framework

Evaluate generated courses using LLM-as-judge with structured rubrics:

```bash
# Install evaluation dependencies
pip3 install -e ".[evaluation]"

# Create dataset from course outputs
python3 -m evaluation.dataset create-dataset --inputs output/*.json

# Run full evaluation on dataset
python3 -m evaluation.workflow evaluate --dataset my-courses

# Quick evaluation of single file
python3 -m evaluation.workflow quick --input output/course.json
```

## ⚙️ Configuration Options

Key configuration parameters in `CourseConfig`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `text_llm_provider` | LLM provider: `mistral`, `gemini`, `groq`, `openai`, `deepseek` | `mistral` |
| `web_search_provider` | Search: `ddg`, `tavily`, `wikipedia` | `ddg` |
| `image_search_provider` | Images: `bing`, `freepik`, `ddg`, `google` | `freepik` |
| `total_pages` | Target course length in pages | `2` |
| `words_per_page` | Words per page target | `400` |
| `language` | Output language | `Español` |
| `enable_research` | Enable research phase | `True` |
| `use_reflection` | Fact verification with web search | `True` |
| `use_vision_ranking` | Pixtral-based image ranking | `False` |
| `select_html` | HTML format selection: `LLM`, `random` | `LLM` |
| `concurrency` | Parallel processing threads | `10` |

## 📁 Output Structure

```
output/CourseName_YYYYMMDD_HHMMSS/
├── course.json          # Complete course data
├── module_0.json        # Individual module files
├── module_1.json
├── bibliography.json    # Book references (if enabled)
├── podcast/             # Audio files and conversations
│   ├── module_0.mp3
│   └── module_0_conversation.json
├── book/                # PDF book output
│   └── course.pdf
└── video_html_generator/  # Simplified JSONs for video
    └── module_0_simplified.json
```

## 🏗️ Project Structure

```
course-generator-agent/
├── agents/                    # AI agent implementations
│   ├── activities_generator/  # Quiz and activity generation
│   ├── bibliography_generator/# Book recommendations
│   ├── html_formatter/        # Interactive HTML elements
│   ├── image_search/          # Image search and selection
│   ├── index_generator/       # Course structure generation
│   ├── pdf_index_generator/   # PDF syllabus extraction
│   ├── podcast_generator/     # Dialogue and TTS synthesis
│   ├── section_theory_generator/  # Content generation
│   └── video_html_generator/  # JSON simplification for video
├── main/                      # Workflow orchestration
│   ├── workflow.py            # Topic-based generation
│   ├── workflow_pdf.py        # PDF-based generation
│   ├── workflow_podcast.py    # Podcast-focused pipeline
│   └── workflow_pdf2podcast.py# PDF to podcast pipeline
├── tools/                     # External integrations
│   ├── websearch/             # DDG, Tavily, Wikipedia
│   ├── imagesearch/           # Bing, DDG, Freepik, Google
│   ├── booksearch/            # Open Library API
│   ├── pdf2md/                # PDF extraction (Docling)
│   ├── json2book/             # LaTeX PDF generation
│   └── podcast/               # TTS engines (Edge, Coqui)
├── evaluators/                # Quality assessment
└── evaluation/                # Evaluation framework
```

## 🔧 Supported Providers

### Text LLMs
- **Mistral**: `mistral-small-latest` (default)
- **Gemini**: `gemini-flash-latest`
- **OpenAI**: `gpt-4o-mini`
- **Groq**: `moonshotai/kimi-k2-instruct-0905`
- **DeepSeek**: `deepseek-chat`

### Vision LLMs
- **Pixtral**: `pixtral-12b-2409`

### TTS Engines
- **Edge TTS**: Cloud-based, no conflicts (default)
- **Coqui TTS**: Local, requires separate environment

## 📄 License

See [LICENSE](LICENSE) for details.
