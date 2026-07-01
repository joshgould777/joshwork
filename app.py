import json
import base64
import boto3
from botocore.config import Config
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure longer timeouts for Bedrock API calls
bedrock_config = Config(
    read_timeout=120,  # 2 minutes read timeout
    connect_timeout=10,
    retries={'max_attempts': 2}
)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# AWS clients
bedrock_agent = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
bedrock_agent_client = boto3.client("bedrock-agent", region_name="us-east-1")
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1", config=bedrock_config)
s3 = boto3.client("s3", region_name="us-east-1")

# Configuration
KNOWLEDGE_BASE_ID = "ASL3UKYBNF"
DATA_SOURCE_ID = "JGZE5LMGIO"
S3_BUCKET = "joshwork-kb-docs-538705487788"
MODEL_ARN = "arn:aws:bedrock:us-east-1:538705487788:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
VISION_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

ALLOWED_EXTENSIONS = {'pdf', 'txt', 'html', 'md', 'csv', 'doc', 'docx', 'json'}
IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp'}

FACIAL_ANALYSIS_PROMPT = """You are a clinical facial aesthetics analyst providing BRUTALLY HONEST assessments based on the golden ratio (phi = 1.618) and objective facial harmony metrics. You have been provided with {photo_description}. The person is {age} years old and weighs {weight} lbs.

## CRITICAL INSTRUCTIONS:
- Be 100% HONEST and ACCURATE. Do NOT soften criticism or spare feelings.
- Rate OBJECTIVELY against golden ratio standards. Most people score 4-6/10 on most features.
- A 10/10 is extremely rare (model-tier). A 7/10 is already above average.
- Do NOT give inflated ratings to be nice. Accuracy is the ONLY goal.
- If something is below average, say it directly. The user NEEDS honest feedback to improve.
- This is a clinical assessment, not a self-esteem exercise.

## Structure your response EXACTLY as follows:

### PROS - Genuine Strengths

Rate each genuinely strong feature on a scale of 1-10 with honest assessment:

**[Feature Name]: [X/10]**
- Why this is objectively a strength
- How it measures against golden ratio standards

(Only list features that GENUINELY score 7/10 or higher. If none qualify, say so.)

---

### CONS - Areas That Need Work

Rate each area on a scale of 1-10 with HONEST assessment and specific fixes:

**[Feature Name]: [X/10]**
- Exactly what is wrong (be specific and direct)
- How far off from ideal proportions
- Specific actionable fixes for age {age}

(List ALL features below 7/10. Be thorough and direct.)

---

### FEATURE-BY-FEATURE RATINGS

Objective ratings for ALL facial focal points. Be HARSH but FAIR:

| Feature | Rating | Honest Assessment |
|---------|--------|-------------------|
| Forehead Proportion | X/10 | Direct assessment |
| Eye Shape & Symmetry | X/10 | Direct assessment |
| Eye Spacing | X/10 | Direct assessment |
| Nose Shape | X/10 | Direct assessment |
| Nose Proportion | X/10 | Direct assessment |
| Lip Shape | X/10 | Direct assessment |
| Lip Ratio (upper:lower) | X/10 | Direct assessment |
| Cheekbone Definition | X/10 | Direct assessment |
| Jaw Definition | X/10 | Direct assessment |
| Chin Projection | X/10 | Direct assessment |
| Facial Symmetry | X/10 | Direct assessment |
| Skin Quality | X/10 | Direct assessment |
| Overall Harmony | X/10 | Direct assessment |

**OVERALL FACIAL SCORE: X/10**

---

### WEIGHT IMPACT ASSESSMENT

At {weight} lbs, evaluate how weight affects facial appearance:
- Would losing weight improve facial definition/jawline?
- Is there excess facial fat obscuring bone structure?
- Would gaining weight help if face is too gaunt?
- Estimate ideal weight range for optimal facial aesthetics

---

### REALISTIC FIXES FOR AGE {age}

**What Can Actually Be Fixed:**
- Non-invasive options (mewing, exercises, skincare, weight loss/gain)
- Minimally invasive (fillers, Botox - be specific about placement)
- Surgical options if warranted (be direct about what would help most)

**What Cannot Be Fixed:**
- Be honest about genetic limitations
- What they should accept vs. what they can change

---

### TOP 3 PRIORITY FIXES
Ranked by impact - what will make the BIGGEST difference:

1. **[Issue]** - Exactly what's wrong and how to fix it
2. **[Issue]** - Exactly what's wrong and how to fix it
3. **[Issue]** - Exactly what's wrong and how to fix it

---

### HARD TRUTHS
State any uncomfortable realities the person needs to hear to make real progress. Do not hold back.

---

### EXPERIMENTAL & EMERGING SOLUTIONS

List cutting-edge or experimental approaches that show promise but aren't yet fully accepted by mainstream science. These are NOT pseudoscience or scams, but legitimate emerging treatments and techniques being researched:

**Emerging Non-Invasive:**
- Mewing / orthotropics (tongue posture for facial development)
- Red light therapy / photobiomodulation for skin
- Microcurrent facial toning devices
- Face yoga / facial exercises with emerging research
- Bone conduction / vibration therapy

**Experimental Medical:**
- PRP (Platelet-Rich Plasma) therapy variations
- Exosome therapy for skin rejuvenation
- Stem cell facial treatments
- Peptide therapies (GHK-Cu, BPC-157, etc.)
- Fat transfer techniques with stem cell enrichment

**Research-Stage:**
- Gene therapy approaches (future)
- Bone remodeling stimulation techniques
- Bioelectric facial sculpting
- Regenerative medicine approaches

For each relevant suggestion, note:
- Current research status
- Potential benefits for THIS person's specific issues
- Risks and unknowns
- Where to find legitimate practitioners/products

**DISCLAIMER:** These are experimental. Results vary. Do your own research. Not FDA approved for these uses.

---

Remember: The user explicitly requested brutal honesty. Sugarcoating does them a disservice. They want to IMPROVE, and that requires knowing EXACTLY what's wrong. Be clinical, be direct, be accurate."""


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        response = bedrock_agent.retrieve_and_generate(
            input={"text": user_message},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn": MODEL_ARN,
                    "generationConfiguration": {
                        "inferenceConfig": {
                            "textInferenceConfig": {
                                "maxTokens": 1024,
                                "temperature": 0.7
                            }
                        }
                    },
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults": 5
                        }
                    }
                }
            }
        )

        assistant_message = response["output"]["text"]

        # Include citations if available
        citations = []
        if "citations" in response:
            for citation in response["citations"]:
                if "retrievedReferences" in citation:
                    for ref in citation["retrievedReferences"]:
                        if "location" in ref and "s3Location" in ref["location"]:
                            uri = ref["location"]["s3Location"].get("uri", "")
                            if uri and uri not in citations:
                                citations.append(uri)

        return jsonify({
            "response": assistant_message,
            "citations": citations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-face", methods=["POST"])
def analyze_face():
    """Analyze face photos for golden ratio proportions and provide recommendations."""
    # Check for front photo (required)
    if "front_image" not in request.files:
        return jsonify({"error": "Front-facing photo is required"}), 400

    front_image = request.files["front_image"]
    if front_image.filename == "":
        return jsonify({"error": "No front image selected"}), 400

    if not allowed_image(front_image.filename):
        return jsonify({"error": f"Image type not allowed. Allowed: {', '.join(IMAGE_EXTENSIONS)}"}), 400

    # Get age (required)
    age = request.form.get("age", "")
    if not age:
        return jsonify({"error": "Age is required for personalized recommendations"}), 400

    try:
        age = int(age)
        if age < 1 or age > 120:
            return jsonify({"error": "Please enter a valid age"}), 400
    except ValueError:
        return jsonify({"error": "Age must be a number"}), 400

    # Get weight (required)
    weight = request.form.get("weight", "")
    if not weight:
        return jsonify({"error": "Weight is required for personalized recommendations"}), 400

    try:
        weight = int(weight)
        if weight < 50 or weight > 500:
            return jsonify({"error": "Please enter a valid weight"}), 400
    except ValueError:
        return jsonify({"error": "Weight must be a number"}), 400

    # Check for side photo (optional but recommended)
    side_image = request.files.get("side_image")
    has_side_photo = side_image and side_image.filename != "" and allowed_image(side_image.filename)

    try:
        # Process front image
        front_data = front_image.read()
        front_base64 = base64.b64encode(front_data).decode("utf-8")
        front_ext = front_image.filename.rsplit('.', 1)[1].lower()

        media_type_map = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        front_media_type = media_type_map.get(front_ext, 'image/jpeg')

        # Build message content with images
        content = [
            {
                "type": "text",
                "text": "FRONT-FACING PHOTO:"
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": front_media_type,
                    "data": front_base64
                }
            }
        ]

        # Add side photo if provided
        photo_description = "a front-facing photo"
        if has_side_photo:
            side_data = side_image.read()
            side_base64 = base64.b64encode(side_data).decode("utf-8")
            side_ext = side_image.filename.rsplit('.', 1)[1].lower()
            side_media_type = media_type_map.get(side_ext, 'image/jpeg')

            content.extend([
                {
                    "type": "text",
                    "text": "SIDE PROFILE PHOTO:"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": side_media_type,
                        "data": side_base64
                    }
                }
            ])
            photo_description = "both a front-facing photo and a side profile photo"

        # Add the analysis prompt
        prompt = FACIAL_ANALYSIS_PROMPT.replace("{photo_description}", photo_description).replace("{age}", str(age)).replace("{weight}", str(weight))
        content.append({
            "type": "text",
            "text": prompt
        })

        # Call Claude with vision
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        })

        response = bedrock_runtime.invoke_model(
            modelId=VISION_MODEL_ID,
            contentType="application/json",
            body=body
        )

        response_body = json.loads(response["body"].read())
        analysis = response_body["content"][0]["text"]

        return jsonify({
            "status": "success",
            "analysis": analysis
        })

    except Exception as e:
        import traceback
        import sys
        error_details = traceback.format_exc()
        print(f"Error in analyze_face: {error_details}", file=sys.stderr)
        print(f"Error in analyze_face: {error_details}", flush=True)
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/generate-optimized", methods=["POST"])
def generate_optimized():
    """Generate an optimized version of the face with suggested improvements applied."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    suggestions_json = request.form.get("suggestions", "[]")
    try:
        suggestions = json.loads(suggestions_json)
    except:
        return jsonify({"error": "Invalid suggestions format"}), 400

    if not suggestions:
        return jsonify({"error": "No suggestions selected"}), 400

    try:
        # Read and encode original image
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Build the modification prompt
        improvements = "\n".join([f"- {s}" for s in suggestions])
        modification_prompt = f"""Transform this face photo to show realistic improvements:

{improvements}

IMPORTANT GUIDELINES:
- Keep the person recognizable - same identity, just improved
- Make subtle, realistic changes - not dramatic transformations
- Maintain natural skin texture and lighting
- Apply changes that would result from the suggested improvements
- Keep the same pose, expression, and background
- Make it look like a realistic "after" photo, not AI-generated"""

        # Use Titan Image Generator V2 for image-to-image transformation
        titan_body = json.dumps({
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": {
                "images": [image_base64],
                "text": modification_prompt,
                "similarityStrength": 0.7
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "width": 1024,
                "height": 1024,
                "cfgScale": 8.0
            }
        })

        try:
            response = bedrock_runtime.invoke_model(
                modelId="amazon.titan-image-generator-v2:0",
                contentType="application/json",
                accept="application/json",
                body=titan_body
            )

            response_body = json.loads(response["body"].read())

            if "images" in response_body and len(response_body["images"]) > 0:
                generated_image = response_body["images"][0]
                return jsonify({
                    "status": "success",
                    "image": generated_image
                })
        except Exception as titan_error:
            print(f"Titan V2 failed: {titan_error}", flush=True)

        # Fallback to Stable Diffusion 3
        try:
            sd3_body = json.dumps({
                "prompt": f"Professional portrait photo showing subtle facial improvements: {', '.join(suggestions[:5])}. Photorealistic, natural lighting, high quality, same person",
                "negative_prompt": "cartoon, anime, drawing, painting, unrealistic, distorted, ugly, different person",
                "image": image_base64,
                "strength": 0.4,
                "mode": "image-to-image",
                "output_format": "png"
            })

            response = bedrock_runtime.invoke_model(
                modelId="stability.sd3-large-v1:0",
                contentType="application/json",
                accept="application/json",
                body=sd3_body
            )

            response_body = json.loads(response["body"].read())

            if "images" in response_body and len(response_body["images"]) > 0:
                generated_image = response_body["images"][0]
                return jsonify({
                    "status": "success",
                    "image": generated_image
                })
        except Exception as sd3_error:
            print(f"SD3 failed: {sd3_error}", flush=True)

        # Final fallback to Stable Diffusion XL
        try:
            sdxl_body = json.dumps({
                "text_prompts": [
                    {
                        "text": f"Professional portrait photo, same person with subtle facial improvements: {', '.join(suggestions[:5])}. Photorealistic, natural lighting, high quality",
                        "weight": 1.0
                    },
                    {
                        "text": "cartoon, anime, drawing, painting, unrealistic, distorted, ugly, different person",
                        "weight": -1.0
                    }
                ],
                "init_image": image_base64,
                "init_image_mode": "IMAGE_STRENGTH",
                "image_strength": 0.35,
                "cfg_scale": 7,
                "samples": 1,
                "steps": 30
            })

            response = bedrock_runtime.invoke_model(
                modelId="stability.stable-diffusion-xl-v1",
                contentType="application/json",
                accept="application/json",
                body=sdxl_body
            )

            response_body = json.loads(response["body"].read())

            if "artifacts" in response_body and len(response_body["artifacts"]) > 0:
                generated_image = response_body["artifacts"][0]["base64"]
                return jsonify({
                    "status": "success",
                    "image": generated_image
                })
        except Exception as sdxl_error:
            print(f"SDXL failed: {sdxl_error}", flush=True)

        return jsonify({"error": "Image generation not available. Please enable Titan Image Generator V2, Stable Diffusion 3, or Stable Diffusion XL in AWS Bedrock Model Access."}), 500

    except Exception as e:
        return jsonify({"error": f"Image generation failed: {str(e)}"}), 500


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    filename = secure_filename(file.filename)

    try:
        # Upload to S3
        s3.upload_fileobj(file, S3_BUCKET, filename)

        return jsonify({
            "status": "uploaded",
            "filename": filename,
            "message": "File uploaded. Click 'Sync Knowledge Base' to index it."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/sync", methods=["POST"])
def sync():
    try:
        response = bedrock_agent_client.start_ingestion_job(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            dataSourceId=DATA_SOURCE_ID
        )

        job_id = response["ingestionJob"]["ingestionJobId"]
        return jsonify({
            "status": "syncing",
            "jobId": job_id,
            "message": "Knowledge base sync started. This may take a few minutes."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/files", methods=["GET"])
def list_files():
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET)
        files = []
        if "Contents" in response:
            for obj in response["Contents"]:
                files.append({
                    "name": obj["Key"],
                    "size": obj["Size"],
                    "modified": obj["LastModified"].isoformat()
                })
        return jsonify({"files": files})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear():
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
