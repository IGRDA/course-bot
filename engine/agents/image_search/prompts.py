from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator, AliasChoices
from typing import List


# ---- Image Query Generation Prompt ----
image_query_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at creating image search queries.
Generate a 3-5 word query that captures the visual concept.
Focus on visual elements that would make a good educational illustration.
Always keep the query relevant to the course topic and the block title and content preview."""),
    
    ("human", """Generate an image search query (3-5 words) for this content block:

Course Topic: {course_title}
Block Title: {block_title}

The image must visually relate to "{course_title}". Return ONLY the search query, nothing else.""")
])


# ---- Image Ranking Schema ----
class ImageRankingScore(BaseModel):
    """Rubric-based score for evaluating an image's suitability for educational content."""
    
    alignment: int = Field(
        ge=0, le=8,
        description="How well image aligns with topic AND block title (0=unrelated, 1=topic only, 2=weak, 4=good, 8=ideal)"
    )
    no_watermark: int = Field(
        ge=0, le=2,
        description="2 if the image has NO watermarks/logos, 0 if it has watermarks"
    )
    has_text: int = Field(
        ge=0,
        le=2,
        validation_alias=AliasChoices("has_text", "no_text"),
        description="0 if image contains text/labels, 2 if clean (no text)"
    )
    total: int = Field(
        ge=0, le=12,
        description="Sum of all scores (alignment + no_watermark + has_text). Max 12 points."
    )
    
    @field_validator('total', mode='before')
    @classmethod
    def compute_total(cls, v, info):
        """Auto-compute total if not provided or validate if provided."""
        data = info.data
        computed = data.get('alignment', 0) + data.get('no_watermark', 0) + data.get('has_text', 0)
        if v is None:
            return computed
        return v


class ImageRankingResult(BaseModel):
    """Result of ranking multiple images."""
    
    scores: List[ImageRankingScore] = Field(
        description="List of scores for each image, in the same order as input images"
    )


# ---- Image Ranking Prompt (for Vision LLM) ----
IMAGE_RANKING_SYSTEM_PROMPT = """You are an expert image evaluator for educational content.
Score each image using this rubric:

SCORING RUBRIC (Max 12 points):

1. ALIGNMENT (0, 1, 2, 4, or 8 points): The image must match BOTH the course topic AND the block title.
   - 0: Unrelated to the course topic
   - 1: Matches course topic only, not the block title
   - 2: Loosely connected to both, but unclear fit
   - 4: Good match for both topic and block title
   - 8: Ideal - perfectly illustrates both the course subject and block concept

2. NO WATERMARKS (0 or 2 points):
   - 2: Clean, no watermarks or logos
   - 0: Has watermarks or stock photo marks

3. HAS_TEXT (0 or 2 points):
   - 2: No visible text/labels (clean image)
   - 0: Contains visible text/labels
   Important: in the JSON output, this field MUST be named "has_text".

Return your evaluation as JSON with the exact structure specified."""


def create_image_ranking_prompt(
    course_title: str,
    block_title: str,
    content_preview: str,
    num_images: int
) -> str:
    """
    Create the human message prompt for image ranking.
    
    Args:
        course_title: The course topic
        block_title: The content block title
        content_preview: Preview of the block content
        num_images: Number of images being evaluated
    
    Returns:
        Formatted prompt string
    """
    return f"""Evaluate {num_images} image(s) for educational content.

CONTEXT:
- Course Topic: {course_title}
- Block Title: {block_title}

Return JSON with this structure:
{{
    "scores": [
        {{
            "alignment": <0|1|2|4|8>,
            "no_watermark": <0|2>,
            "has_text": <0|2>,
            "total": <sum>
        }}
    ]
}}

Array must have exactly {num_images} element(s). Return ONLY JSON."""


def create_single_image_ranking_prompt(
    course_title: str,
    block_title: str,
    content_preview: str,
) -> str:
    """
    Create the human message prompt for scoring a single image.
    
    Args:
        course_title: The course topic
        block_title: The content block title
        content_preview: Preview of the block content
    
    Returns:
        Formatted prompt string for single image evaluation
    """
    return f"""Evaluate this image for educational content.

CONTEXT:
- Course Topic: {course_title}
- Block Title: {block_title}

The image must match BOTH the topic AND the block title.
Return JSON:
{{
    "alignment": <0|1|2|4|8>,
    "no_watermark": <0|2>,
    "has_text": <0|2>,
    "total": <sum>
}}

Return ONLY JSON."""


# ---- Fallback Parsing Prompt ----
FALLBACK_PARSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a JSON parser. Extract structured data from text and return valid JSON.
If values are missing, use defaults: alignment=0, no_watermark=0, has_text=0, total=0."""),
    
    ("human", """Parse this text and extract image ranking scores.

Expected JSON:
{{
    "scores": [
        {{
            "alignment": <0|1|2|4|8>,
            "no_watermark": <0|2>,
            "has_text": <0|2>,
            "total": <sum>
        }}
    ]
}}

Number of images: {num_images}

Text to parse:
{raw_text}

Return ONLY valid JSON.""")
])
