#!/usr/bin/env python3
"""
Gemini Batch API Image Generator

Uses Google's Gemini Batch API to generate images from text prompts.
The Batch API offers 50% cost savings compared to the standard API.

Documentation: https://ai.google.dev/gemini-api/docs/batch-api?batch=file#image-generation
"""

import base64
import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types


# Configuration
# Model for batch image generation (per Google docs)
# See: https://ai.google.dev/gemini-api/docs/batch-api?batch=file#image-generation
MODEL_NAME = "gemini-2.5-flash-image"  # gemini-3-pro-image-preview
# Output to project root's output/generated_images folder
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "generated_images"
POLL_INTERVAL_SECONDS = 10

# Image generation prompts
IMAGE_PROMPTS = [
    #{"key": "request-1-flash", "request": {"contents": [{"parts": [{"text": "sketch of photoelectric effect"}]}], "generation_config": {"responseModalities": ["IMAGE"]}}},
    #{"key": "request-2-flash", "request": {"contents": [{"parts": [{"text": "population pyramid in spain"}]}], "generation_config": {"responseModalities": ["IMAGE"]}}},
    #{"key": "request-3-flash", "request": {"contents": [{"parts": [{"text": "Early quantum physics scientists timeline"}]}], "generation_config": {"responseModalities": ["IMAGE"]}}},
    #{"key": "request-4-flash", "request": {"contents": [{"parts": [{"text": "Double-slit electron interference pattern"}]}], "generation_config": {"responseModalities": ["IMAGE"]}}},
    #{"key": "request-5-flash", "request": {"contents": [{"parts": [{"text": "Ec = hν − Φ graph"}]}], "generation_config": {"responseModalities": ["IMAGE"]}}},
    {"key": "request-6-flash", "request": {"contents": [{"parts": [{"text": "Atomic discrete energy levels and the Schrödinger equation"}]}], "generation_config": {"responseModalities": ["IMAGE"]}}},
    #{"key": "request-7-flash", "request": {"contents": [{"parts": [{"text": "Discrete photons in light"}]}], "generation_config": {"responseModalities": ["IMAGE"]}}},


    ]


def create_jsonl_file(prompts: list[dict], output_path: Path) -> Path:
    """Create a JSONL file from the prompts list."""
    with open(output_path, "w") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")
    print(f"✅ Created JSONL file: {output_path}")
    return output_path


def upload_jsonl_file(client: genai.Client, file_path: Path) -> str:
    """Upload the JSONL file to the File API."""
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


def create_batch_job(client: genai.Client, file_name: str, model: str) -> str:
    """Create a batch job for image generation."""
    print(f"🚀 Creating batch job with model: {model}...")
    batch_job = client.batches.create(
        model=f"models/{model}",
        src=file_name,
        config={
            "display_name": "image-generation-batch-job",
        },
    )
    print(f"✅ Created batch job: {batch_job.name}")
    return batch_job.name


def wait_for_completion(client: genai.Client, job_name: str) -> dict:
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


def save_images_from_results(client: genai.Client, batch_job: dict, output_dir: Path) -> list[Path]:
    """Download and save images from the batch job results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    
    if batch_job.state != "JOB_STATE_SUCCEEDED":
        print(f"❌ Job did not succeed. State: {batch_job.state}")
        if hasattr(batch_job, 'error') and batch_job.error:
            print(f"   Error: {batch_job.error}")
        return saved_paths
    
    # Check if results are inline or in a file
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
                saved_path = extract_and_save_image(result, output_dir)
                if saved_path:
                    saved_paths.append(saved_path)
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse result line: {e}")
    
    elif hasattr(batch_job, 'response') and batch_job.response:
        # Inline responses
        print("📥 Processing inline responses...")
        for idx, response in enumerate(batch_job.response.responses):
            result = {"key": f"request-{idx + 1}", "response": response}
            saved_path = extract_and_save_image(result, output_dir)
            if saved_path:
                saved_paths.append(saved_path)
    
    return saved_paths


def extract_and_save_image(result: dict, output_dir: Path) -> Path | None:
    """Extract image data from a result and save it to disk."""
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
    
    for part_idx, part in enumerate(parts):
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
            
            # Create filename
            filename = f"{key}{f'_part{part_idx}' if len(parts) > 1 else ''}{ext}"
            output_path = output_dir / filename
            
            # Decode and save
            try:
                image_bytes = base64.b64decode(image_data)
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                print(f"💾 Saved: {output_path}")
                return output_path
            except Exception as e:
                print(f"⚠️ Failed to save image for {key}: {e}")
                return None
        
        elif "text" in part:
            # Some responses may include text descriptions
            print(f"📝 Text response for {key}: {part['text'][:100]}...")
    
    print(f"⚠️ No image data found for {key}")
    return None


def main():
    """Main function to run the batch image generation."""
    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("   Set it with: export GEMINI_API_KEY='your-api-key'")
        return
    
    print("🎨 Gemini Batch API Image Generator")
    print("=" * 50)
    print(f"Model: {MODEL_NAME}")
    print(f"Number of prompts: {len(IMAGE_PROMPTS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Initialize client
    client = genai.Client(api_key=api_key)
    
    # Create JSONL file with prompts
    jsonl_path = Path(__file__).parent / "batch_requests.jsonl"
    create_jsonl_file(IMAGE_PROMPTS, jsonl_path)
    
    # Upload the file
    file_name = upload_jsonl_file(client, jsonl_path)
    
    # Create batch job
    job_name = create_batch_job(client, file_name, MODEL_NAME)
    
    # Wait for completion
    batch_job = wait_for_completion(client, job_name)
    
    # Save images
    print()
    saved_paths = save_images_from_results(client, batch_job, OUTPUT_DIR)
    
    # Summary
    print()
    print("=" * 50)
    print(f"✨ Completed! Saved {len(saved_paths)} images to {OUTPUT_DIR}")
    
    # Cleanup temporary JSONL file
    if jsonl_path.exists():
        jsonl_path.unlink()
        print(f"🧹 Cleaned up temporary file: {jsonl_path}")


if __name__ == "__main__":
    main()

