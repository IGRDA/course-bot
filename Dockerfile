FROM python:3.12-slim

# =============================================================================
# Layer 1: System packages + Node.js + LaTeX
# This is the most stable layer. Rarely changes.
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git procps \
    # Playwright/Chromium runtime dependencies
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 \
    libxdamage1 libxfixes3 libxrandr2 libgbm1 \
    libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 \
    libwayland-client0 \
    # Docling / OpenCV headless deps
    libgl1 libglib2.0-0 libgomp1 \
    # LaTeX for PDF book generation (xetex for native Unicode support)
    texlive-xetex \
    texlive-latex-base texlive-latex-extra \
    texlive-fonts-recommended texlive-fonts-extra \
    texlive-bibtex-extra biber \
    texlive-lang-spanish \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# =============================================================================
# Layer 2: Engine Python dependencies (changes when engine/pyproject.toml changes)
# =============================================================================
COPY engine/pyproject.toml /app/engine/pyproject.toml
RUN mkdir -p /app/engine/agents /app/engine/workflows /app/engine/tools \
             /app/engine/evaluators /app/engine/evaluation /app/engine/LLMs && \
    touch /app/engine/agents/__init__.py /app/engine/workflows/__init__.py \
          /app/engine/tools/__init__.py /app/engine/evaluators/__init__.py \
          /app/engine/evaluation/__init__.py /app/engine/LLMs/__init__.py && \
    pip install --no-cache-dir -e "/app/engine[pdf-extraction]" && \
    pip install --no-cache-dir pymupdf

# =============================================================================
# Layer 3: Playwright Chromium browser (~300 MB, changes rarely)
# Installed after engine deps so playwright version matches pyproject.toml pin.
# =============================================================================
RUN playwright install chromium

# =============================================================================
# Layer 4: Pre-downloaded models (changes rarely — before bot deps so bot
# pyproject.toml changes don't re-download ~1 GB of OCR models)
# =============================================================================
RUN python -c "import easyocr; easyocr.Reader(['es','en'], gpu=False)"

# Pre-download Docling layout + table-structure models so the converter
# works offline inside Cloud Run (no HuggingFace access at runtime).
RUN python -c "\
from docling.models.layout_model import LayoutModel; \
from docling.models.table_structure_model import TableStructureModel; \
LayoutModel.download_models(progress=True); \
TableStructureModel.download_models(progress=True)"

# =============================================================================
# Layer 5: Bot Python dependencies (changes when bot/pyproject.toml changes)
# =============================================================================
COPY bot/pyproject.toml /app/bot/pyproject.toml
RUN mkdir -p /app/bot && \
    touch /app/bot/__init__.py && \
    pip install --no-cache-dir -e /app/bot

# =============================================================================
# Layer 6: Git identity + application code
# =============================================================================
RUN git config --global user.email "course-bot@noreply" && \
    git config --global user.name "course-bot"

COPY engine/ /app/engine/
COPY bot/ /app/bot/

CMD ["uvicorn", "bot.main:app", "--host", "0.0.0.0", "--port", "8080"]
