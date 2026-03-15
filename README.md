# Course Bot

A Slack bot that generates complete educational courses using AI. Deployed on **Google Cloud Run** with **Terraform** infrastructure-as-code. Course generation is powered by **Claude** via the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python), which orchestrates a LangGraph-based course engine.

When someone mentions the bot in Slack (`@course-bot`), Claude reads the course engine's instructions, runs the generation workflow, and replies with the result. The bot also responds to direct messages and can process attached files (PDFs, documents, etc.).

## Architecture

```
  Slack user tags @course-bot
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  API Server (Cloud Run Service) — bot/                   │
│                                                          │
│  POST /slack/events                                      │
│    ├─ Verify Slack signature                             │
│    ├─ Parse event, run filters/acknowledgers             │
│    └─ Dispatch to Cloud Run Job (heavy work)             │
│       or handle in-process (reactions, joins)            │
└───────────────────────┬──────────────────────────────────┘
                        │ Cloud Run Job execution
                        ▼
┌──────────────────────────────────────────────────────────┐
│  Worker (Cloud Run Job) — bot/job/worker.py              │
│                                                          │
│  1. Decode event payload                                 │
│  2. Clone monorepo → /tmp/session-<hash>/                │
│  3. Set Claude cwd = /tmp/session-<hash>/engine/         │
│  4. Claude reads engine/CLAUDE.md + engine/.claude/      │
│  5. Claude runs: python3 -m workflows.workflow           │
│  6. Bot zips engine/output/ and uploads to Slack thread  │
└──────────────────────────────────────────────────────────┘
```

The bot and engine are fully decoupled. Claude never leaves `engine/`. The bot never enters `engine/`. Each Slack thread gets its own isolated git workspace.

## Project Structure

```
course-bot/
├── engine/                           # Course generation engine (Claude's world)
│   ├── agents/                       # AI agent implementations
│   │   ├── activities_generator/     # Quiz and activity generation
│   │   ├── bibliography_generator/   # Book recommendations (Open Library)
│   │   ├── html_formatter/           # Interactive HTML elements
│   │   ├── image_search/             # Image search and selection
│   │   ├── index_generator/          # Course structure generation
│   │   ├── pdf_index_generator/      # PDF syllabus extraction
│   │   ├── podcast_generator/        # Dialogue and TTS synthesis
│   │   ├── section_theory_generator/ # Content generation
│   │   ├── video_html_generator/     # JSON simplification for video
│   │   └── ...
│   ├── workflows/                    # LangGraph workflow orchestration
│   │   ├── workflow.py               # Topic-based generation
│   │   ├── workflow_pdf.py           # PDF-based generation
│   │   ├── workflow_podcast.py       # Podcast-focused pipeline
│   │   ├── workflow_digitalize.py    # Digitalize existing content
│   │   └── ...
│   ├── tools/                        # External integrations
│   │   ├── websearch/                # DDG, Tavily, Wikipedia, Scholar
│   │   ├── imagesearch/              # Bing, DDG, Freepik, Google
│   │   ├── booksearch/               # Google Books, Open Library
│   │   ├── pdf2md/                   # PDF extraction (Docling)
│   │   ├── json2book/                # LaTeX PDF book generation
│   │   └── podcast/                  # TTS engines (Edge, Coqui, ElevenLabs)
│   ├── LLMs/                         # LLM provider abstraction
│   ├── .claude/skills/               # Claude Code skills
│   ├── CLAUDE.md                     # Claude's instructions
│   ├── pyproject.toml                # Engine package definition
│   └── env.example                   # Engine-specific env vars
│
├── bot/                              # Slack application
│   ├── main.py                       # FastAPI entry point
│   ├── config.py                     # Environment variable config
│   ├── webhook/                      # Generic webhook framework
│   ├── slack/                        # Slack-specific layer (signature, events, filters)
│   ├── internal/                     # Business logic (ClaudeClient, SlackService)
│   ├── conversation/                 # Conversation handling
│   ├── workspace/                    # Per-thread git workspace isolation
│   ├── job/                          # Cloud Run Job dispatcher + worker
│   ├── server/                       # HTTP server with graceful shutdown
│   ├── tests/                        # Bot test suite
│   └── pyproject.toml                # Bot package definition
│
├── terraform/                        # Infrastructure as Code
│   ├── main.tf                       # Provider + backend config
│   ├── variables.tf                  # Input variables
│   ├── cloud_run.tf                  # Cloud Run service + job
│   ├── iam.tf                        # Service account + secrets
│   ├── backends/dev.conf             # Remote state config
│   └── env/dev.tfvars                # Dev environment variables
│
├── scripts/deploy.sh                 # Build, push, and deploy script
├── Dockerfile                        # Optimized multi-layer build
└── .github/workflows/ci.yml          # CI pipeline
```

## Prerequisites

- Python 3.12+
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`)
- [Terraform](https://developer.hashicorp.com/terraform/downloads) >= 1.5
- [Docker](https://docs.docker.com/get-docker/) (with `--platform linux/amd64` support)
- A Slack workspace where you can create apps

## Setup

### 1. Create a Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and create a new app
2. Under **OAuth & Permissions**, add bot token scopes:
   - `app_mentions:read`, `channels:history`, `chat:write`, `files:read`, `files:write`
   - `groups:history`, `im:history`, `reactions:read`, `reactions:write`, `users:read`
3. Install the app to your workspace
4. Save the **Bot User OAuth Token** (`xoxb-...`) and **Signing Secret**

### 2. Authenticate with GCP

```bash
gcloud config set project course-bot-490222
gcloud auth configure-docker
gcloud services enable run.googleapis.com containerregistry.googleapis.com \
  iam.googleapis.com serviceusage.googleapis.com secretmanager.googleapis.com
```

### 3. Set Secrets in Secret Manager

The bot requires these secrets (managed via GCP Secret Manager, referenced in Terraform):

| Secret | Description |
|--------|-------------|
| `slack-signing-secret` | Slack request verification |
| `slack-bot-token` | Slack API access (`xoxb-...`) |
| `claude-code-oauth-token` | Anthropic OAuth token for Claude Code |
| `mistral-api-key` | Mistral API key (for course generation) |
| `google-books-api-key` | Google Books API key (bibliography) |
| `youtube-api-key` | YouTube Data API key (video search) |

### 4. Build and Deploy

```bash
# One command: build image, push to GCR, terraform plan + apply
./scripts/deploy.sh dev --build

# Or infrastructure only (image already pushed)
./scripts/deploy.sh dev
```

Manual steps:

```bash
docker build --platform linux/amd64 -t gcr.io/course-bot-490222/course-bot:latest .
docker push gcr.io/course-bot-490222/course-bot:latest

cd terraform
terraform init -backend-config=backends/dev.conf
terraform plan -var-file=env/dev.tfvars
terraform apply -var-file=env/dev.tfvars
```

### 5. Configure Slack Event Subscriptions

1. In Slack app settings, go to **Event Subscriptions** and enable events
2. Set **Request URL** to: `{cloud_run_url}/slack/events`
3. Subscribe to bot events: `app_mention`, `message.im`, `member_joined_channel`, `reaction_added`, `reaction_removed`

## Local Development

### Engine (standalone)

The course engine can be used independently without the Slack bot:

```bash
cd engine
pip install -e .
playwright install chromium

# Configure API keys
cp env.example env.secrets
# Edit env.secrets with your keys
source env.secrets

# Generate a course from a topic
python3 -m workflows.workflow --total-pages 5

# Generate from PDF syllabus
python3 -m workflows.workflow_pdf

# Generate podcast
python3 -m workflows.workflow_podcast --total-pages 5 --tts-engine edge

# Generate PDF book from course output
python3 -m tools.json2book output/CourseName/course.json
```

Optional heavy dependency groups:

```bash
pip install -e ".[pdf-extraction]"    # Docling + EasyOCR (~2 GB)
pip install -e ".[ml]"                # Torch + Transformers (~2 GB)
pip install -e ".[evaluation]"        # Course quality evaluation
```

### Bot (with local Slack)

```bash
cd bot
pip install -e .

export SLACK_SIGNING_SECRET="your-secret"
export SLACK_BOT_TOKEN="xoxb-your-token"
export GCP_PROJECT_ID="course-bot-490222"

uvicorn bot.main:app --port 8080 --reload
```

Use [ngrok](https://ngrok.com/) to expose port 8080 for Slack event delivery:

```bash
ngrok http 8080
```

## Configuration

### Bot Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SLACK_SIGNING_SECRET` | Yes | -- | Slack signing secret |
| `SLACK_BOT_TOKEN` | Yes | -- | Slack bot token (`xoxb-...`) |
| `GCP_PROJECT_ID` | Yes | -- | GCP project ID |
| `GCP_REGION` | No | `europe-west1` | GCP region |
| `JOB_NAME` | No | `course-bot-worker` | Cloud Run Job name |
| `MAX_CONCURRENT_JOBS` | No | `5` | Max concurrent job executions |
| `PORT` | No | `8080` | HTTP server port |
| `LOG_LEVEL` | No | `INFO` | Logging level |

### Worker Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CLAUDE_AGENT_DIR` | No | `/app/engine` | Path to engine directory |
| `CLAUDE_CODE_OAUTH_TOKEN` | Yes | -- | Claude OAuth token |
| `CLAUDE_RESPONSE_TIMEOUT` | No | `5400` | Claude timeout (seconds) |
| `REPO_URL` | No | -- | Git repo URL for workspace isolation |
| `WORKSPACE_BASE_DIR` | No | `/tmp` | Base directory for workspaces |

### Engine Environment Variables

See [`engine/env.example`](engine/env.example) for the full list of LLM provider keys, search API keys, and configuration options.

## Engine Features

- **Multiple Input Sources**: Topics, PDF syllabi, URLs, or existing markdown
- **Complete Pipeline**: Research, Index, Theory, Activities, HTML, Images, Bibliography
- **Multi-Provider LLM**: Mistral, Gemini, OpenAI, Groq, DeepSeek
- **Podcast Generation**: Two-speaker dialogue with Edge TTS, Coqui, or ElevenLabs
- **PDF Book Export**: LaTeX-based professional PDF generation
- **Vision-Powered Image Selection**: Optional Pixtral-based ranking
- **Evaluation Framework**: LLM-as-judge quality assessment
- **LangSmith Tracing**: Full observability for debugging

## Supported Providers

- **Text LLMs**: Mistral (`mistral-small-latest`), Gemini (`gemini-flash-latest`), OpenAI (`gpt-4o-mini`), Groq, DeepSeek
- **Vision**: Pixtral (`pixtral-12b-2409`)
- **TTS**: Edge TTS (default, cloud), Coqui (local), ElevenLabs (cloud)
- **Search**: DuckDuckGo (default), Tavily, Wikipedia, Google Scholar

## License

See [LICENSE](LICENSE) for details.
