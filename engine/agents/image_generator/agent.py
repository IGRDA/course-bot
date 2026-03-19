"""
Image Generator Agent using Gemini Batch API.

This module provides functionality to:
1. Extract image queries from module JSON files
2. Generate images using Gemini's Batch API
3. Save images to a local folder
4. Update the module JSON with base64-encoded images

The Batch API offers 50% cost savings compared to the standard API.
"""

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google import genai
    from google.genai import types


# Configuration
DEFAULT_MODEL = "gemini-3-pro-image-preview"
POLL_INTERVAL_SECONDS = 10
INTERACTIVE_FORMATS = [
    "paragraphs", "accordion", "tabs", "carousel", 
    "flip", "timeline", "conversation"
]


@dataclass
class ImageLocation:
    """Tracks the location of an image within the module JSON structure."""
    submodule_idx: int
    section_idx: int
    element_idx: int
    block_idx: int
    query: str
    request_key: str  # Unique key for batch request matching


def extract_image_queries(module_data: dict) -> list[ImageLocation]:
    """
    Extract all image queries from a module JSON structure.
    
    Traverses the module structure to find all ParagraphBlock.image.query fields
    within interactive HTML elements (paragraphs, accordion, tabs, etc.).
    
    Args:
        module_data: Parsed module JSON dictionary
        
    Returns:
        List of ImageLocation objects with query and location info
    """
    locations: list[ImageLocation] = []
    request_idx = 0
    
    submodules = module_data.get("submodules", [])
    
    for sm_idx, submodule in enumerate(submodules):
        sections = submodule.get("sections", [])
        
        for s_idx, section in enumerate(sections):
            html_elements = section.get("html", []) or []
            
            for e_idx, element in enumerate(html_elements):
                element_type = element.get("type", "")
                
                if element_type in INTERACTIVE_FORMATS:
                    content = element.get("content", [])
                    
                    if isinstance(content, list):
                        for b_idx, block in enumerate(content):
                            image = block.get("image")
                            
                            if image and isinstance(image, dict):
                                query = image.get("query", "")
                                
                                if query:
                                    locations.append(ImageLocation(
                                        submodule_idx=sm_idx,
                                        section_idx=s_idx,
                                        element_idx=e_idx,
                                        block_idx=b_idx,
                                        query=query,
                                        request_key=f"image-{request_idx:04d}"
                                    ))
                                    request_idx += 1
    
    return locations


def create_batch_requests(locations: list[ImageLocation]) -> list[dict]:
    """
    Create JSONL-compatible batch requests for Gemini image generation.
    
    Args:
        locations: List of ImageLocation objects with queries
        
    Returns:
        List of request dictionaries for JSONL serialization
    """
    requests = []
    
    for loc in locations:
        request = {
            "key": loc.request_key,
            "request": {
                "contents": [
                    {
                        "parts": [
                            {"text": loc.query}
                        ]
                    }
                ],
                "generation_config": {
                    "responseModalities": ["IMAGE"]
                }
            }
        }
        requests.append(request)
    
    return requests


def create_jsonl_file(requests: list[dict], output_path: Path) -> Path:
    """Create a JSONL file from the requests list."""
    with open(output_path, "w") as f:
        for request in requests:
            f.write(json.dumps(request) + "\n")
    print(f"✅ Created JSONL file: {output_path}")
    return output_path


def upload_jsonl_file(client, file_path: Path) -> str:
    """Upload the JSONL file to the Gemini File API."""
    from google.genai import types

    print(f"📤 Uploading file: {file_path.name}...")
    uploaded_file = client.files.upload(
        file=str(file_path),
        config=types.UploadFileConfig(
            display_name=file_path.stem,
            mime_type="application/jsonl"
        )
    )
    print(f"✅ Uploaded file: {uploaded_file.name}")
    return uploaded_file.name


def create_batch_job(client, file_name: str, model: str) -> str:
    """Create a batch job for image generation."""
    print(f"🚀 Creating batch job with model: {model}...")
    batch_job = client.batches.create(
        model=f"models/{model}",
        src=file_name,
        config={
            "display_name": "module-image-generation",
        },
    )
    print(f"✅ Created batch job: {batch_job.name}")
    return batch_job.name


def wait_for_completion(client, job_name: str) -> dict:
    """Poll the batch job until it completes."""
    completed_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
    
    print(f"⏳ Waiting for batch job to complete...")
    while True:
        batch_job = client.batches.get(name=job_name)
        state = batch_job.state
        print(f"   Current state: {state}")
        
        if state in completed_states:
            return batch_job
        
        time.sleep(POLL_INTERVAL_SECONDS)


@dataclass
class GeneratedImage:
    """Holds a generated image with its metadata."""
    request_key: str
    image_bytes: bytes
    mime_type: str
    extension: str


def extract_images_from_results(
    client, 
    batch_job: dict
) -> dict[str, GeneratedImage]:
    """
    Extract images from batch job results.
    
    Args:
        client: Gemini client
        batch_job: Completed batch job object
        
    Returns:
        Dictionary mapping request_key to GeneratedImage
    """
    images: dict[str, GeneratedImage] = {}
    
    if batch_job.state != "JOB_STATE_SUCCEEDED":
        print(f"❌ Job did not succeed. State: {batch_job.state}")
        if hasattr(batch_job, 'error') and batch_job.error:
            print(f"   Error: {batch_job.error}")
        return images
    
    # Check if results are in a file
    if hasattr(batch_job, 'dest') and batch_job.dest and hasattr(batch_job.dest, 'file_name'):
        result_file_name = batch_job.dest.file_name
        print(f"📥 Downloading results from: {result_file_name}")
        
        # Download the result file
        file_content = client.files.download(file=result_file_name)
        content_str = file_content.decode('utf-8') if isinstance(file_content, bytes) else str(file_content)
        
        # Parse each line of the JSONL response
        for line in content_str.strip().split('\n'):
            if not line:
                continue
            
            try:
                result = json.loads(line)
                image = extract_image_from_result(result)
                if image:
                    images[image.request_key] = image
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse result line: {e}")
    
    elif hasattr(batch_job, 'response') and batch_job.response:
        # Inline responses
        print("📥 Processing inline responses...")
        for idx, response in enumerate(batch_job.response.responses):
            result = {"key": f"image-{idx:04d}", "response": response}
            image = extract_image_from_result(result)
            if image:
                images[image.request_key] = image
    
    return images


def extract_image_from_result(result: dict) -> GeneratedImage | None:
    """Extract image data from a single result."""
    key = result.get("key", "unknown")
    
    if "error" in result:
        print(f"⚠️ Error for {key}: {result['error']}")
        return None
    
    response = result.get("response", {})
    candidates = response.get("candidates", [])
    
    if not candidates:
        print(f"⚠️ No candidates for {key}")
        return None
    
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    
    for part in parts:
        if "inlineData" in part:
            inline_data = part["inlineData"]
            mime_type = inline_data.get("mimeType", "image/png")
            image_data = inline_data.get("data", "")
            
            # Determine file extension from mime type
            ext_map = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/webp": ".webp",
                "image/gif": ".gif",
            }
            ext = ext_map.get(mime_type, ".png")
            
            try:
                image_bytes = base64.b64decode(image_data)
                return GeneratedImage(
                    request_key=key,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    extension=ext
                )
            except Exception as e:
                print(f"⚠️ Failed to decode image for {key}: {e}")
                return None
        
        elif "text" in part:
            # Some responses may include text descriptions
            print(f"📝 Text response for {key}: {part['text'][:100]}...")
    
    print(f"⚠️ No image data found for {key}")
    return None


def save_images_to_folder(
    images: dict[str, GeneratedImage],
    locations: list[ImageLocation],
    output_dir: Path
) -> dict[str, Path]:
    """
    Save generated images to the output folder.
    
    Args:
        images: Dictionary of request_key to GeneratedImage
        locations: Original image locations for naming
        output_dir: Directory to save images
        
    Returns:
        Dictionary mapping request_key to saved file path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}
    
    for loc in locations:
        if loc.request_key in images:
            img = images[loc.request_key]
            
            # Create descriptive filename
            filename = f"{loc.request_key}{img.extension}"
            output_path = output_dir / filename
            
            with open(output_path, "wb") as f:
                f.write(img.image_bytes)
            
            saved_paths[loc.request_key] = output_path
            print(f"💾 Saved: {output_path.name}")
    
    return saved_paths


def update_module_with_images(
    module_data: dict,
    images: dict[str, GeneratedImage],
    locations: list[ImageLocation]
) -> dict:
    """
    Update the module JSON with base64-encoded images.
    
    Args:
        module_data: Original module JSON data
        images: Dictionary of request_key to GeneratedImage
        locations: Image locations in the module structure
        
    Returns:
        Updated module data with base64 images
    """
    for loc in locations:
        if loc.request_key not in images:
            continue
        
        img = images[loc.request_key]
        
        # Navigate to the image location
        try:
            submodule = module_data["submodules"][loc.submodule_idx]
            section = submodule["sections"][loc.section_idx]
            element = section["html"][loc.element_idx]
            block = element["content"][loc.block_idx]
            
            # Update the image content with base64 data
            mime_type = img.mime_type
            b64_data = base64.b64encode(img.image_bytes).decode('utf-8')
            
            block["image"]["content"] = f"data:{mime_type};base64,{b64_data}"
            
        except (KeyError, IndexError) as e:
            print(f"⚠️ Failed to update image at {loc.request_key}: {e}")
    
    return module_data


def generate_images_batch(
    locations: list[ImageLocation],
    api_key: str | None = None,
    model: str = DEFAULT_MODEL
) -> dict[str, GeneratedImage]:
    """
    Generate images using Gemini Batch API.
    
    Args:
        locations: List of ImageLocation objects with queries
        api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        model: Gemini model to use for image generation
        
    Returns:
        Dictionary mapping request_key to GeneratedImage
    """
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Set it with: export GEMINI_API_KEY='your-api-key'"
        )
    
    if not locations:
        print("⚠️ No image queries to process")
        return {}
    
    print(f"🎨 Generating {len(locations)} images with Gemini Batch API")
    print(f"   Model: {model}")
    
    # Lazy import of google genai SDK
    from google import genai

    # Initialize client
    client = genai.Client(api_key=api_key)
    
    # Create batch requests
    requests = create_batch_requests(locations)
    
    # Create temporary JSONL file
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.jsonl', 
        delete=False
    ) as f:
        jsonl_path = Path(f.name)
    
    try:
        # Write JSONL
        create_jsonl_file(requests, jsonl_path)
        
        # Upload file
        file_name = upload_jsonl_file(client, jsonl_path)
        
        # Create batch job
        job_name = create_batch_job(client, file_name, model)
        
        # Wait for completion
        batch_job = wait_for_completion(client, job_name)
        
        # Extract images
        images = extract_images_from_results(client, batch_job)
        
        print(f"✅ Generated {len(images)}/{len(locations)} images")
        
        return images
        
    finally:
        # Cleanup temporary file
        if jsonl_path.exists():
            jsonl_path.unlink()


def process_module_images(
    module_path: str | Path,
    output_json_path: str | Path | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False
) -> tuple[dict, Path]:
    """
    Process a module JSON file: generate images and update the file.
    
    This is the main entry point for the image generator agent.
    
    Args:
        module_path: Path to the module JSON file
        output_json_path: Path for output JSON (defaults to overwriting input)
        api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        model: Gemini model for image generation
        dry_run: If True, only extract queries without generating images
        
    Returns:
        Tuple of (updated module data, images folder path)
    """
    module_path = Path(module_path)
    
    if not module_path.exists():
        raise FileNotFoundError(f"Module file not found: {module_path}")
    
    # Determine output paths
    if output_json_path:
        output_json_path = Path(output_json_path)
    else:
        output_json_path = module_path
    
    images_dir = module_path.parent / "images"
    
    print("=" * 60)
    print("🖼️  Gemini Image Generator Agent")
    print("=" * 60)
    print(f"Input:  {module_path}")
    print(f"Output: {output_json_path}")
    print(f"Images: {images_dir}")
    print(f"Model:  {model}")
    print()
    
    # Load module JSON
    print("📂 Loading module JSON...")
    with open(module_path, 'r', encoding='utf-8') as f:
        module_data = json.load(f)
    
    # Extract image queries
    print("🔍 Extracting image queries...")
    locations = extract_image_queries(module_data)
    print(f"   Found {len(locations)} images with queries")
    
    if not locations:
        print("⚠️ No images with queries found in module")
        return module_data, images_dir
    
    # Print queries
    print("\n📝 Image queries:")
    for i, loc in enumerate(locations, 1):
        print(f"   {i}. {loc.query[:60]}{'...' if len(loc.query) > 60 else ''}")
    print()
    
    if dry_run:
        print("🔄 Dry run mode - skipping image generation")
        return module_data, images_dir
    
    # Generate images via batch API
    images = generate_images_batch(locations, api_key=api_key, model=model)
    
    if not images:
        print("❌ No images were generated")
        return module_data, images_dir
    
    # Save images to folder
    print(f"\n💾 Saving images to {images_dir}...")
    saved_paths = save_images_to_folder(images, locations, images_dir)
    
    # Update module JSON with base64 images
    print("\n📝 Updating module JSON with base64 images...")
    updated_module = update_module_with_images(module_data, images, locations)
    
    # Save updated JSON
    print(f"\n💾 Saving updated JSON to {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(updated_module, f, indent=2, ensure_ascii=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("✨ Summary")
    print("=" * 60)
    print(f"   Total queries: {len(locations)}")
    print(f"   Images generated: {len(images)}")
    print(f"   Images saved: {len(saved_paths)}")
    print(f"   Output JSON: {output_json_path}")
    print(f"   Images folder: {images_dir}")
    
    return updated_module, images_dir




