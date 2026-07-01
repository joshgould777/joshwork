import json
import base64
import boto3
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# AWS clients
bedrock_agent = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
bedrock_agent_client = boto3.client("bedrock-agent", region_name="us-east-1")
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
s3 = boto3.client("s3", region_name="us-east-1")

# Configuration
KNOWLEDGE_BASE_ID = "ASL3UKYBNF"
DATA_SOURCE_ID = "JGZE5LMGIO"
S3_BUCKET = "joshwork-kb-docs-538705487788"
MODEL_ARN = "arn:aws:bedrock:us-east-1:538705487788:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
VISION_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

ALLOWED_EXTENSIONS = {'pdf', 'txt', 'html', 'md', 'csv', 'doc', 'docx', 'json'}
IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp'}

FACIAL_ANALYSIS_PROMPT = """You are an expert facial aesthetics analyst specializing in the golden ratio (phi = 1.618) and facial harmony. You have been provided with {photo_description}. The person is {age} years old.

## IMPORTANT: Structure your response EXACTLY as follows:

### PROS - Your Strengths (Features That Excel)

Rate each strong feature on a scale of 1-10 and explain why it's a strength:

**[Feature Name]: [X/10]**
- Description of why this feature is strong
- How it contributes to overall harmony

(List all features that score 7/10 or higher)

---

### CONS - Areas for Potential Enhancement

Rate each area on a scale of 1-10 (lower = more room for improvement) and provide age-appropriate recommendations:

**[Feature Name]: [X/10]**
- What could be improved
- Age-appropriate recommendations (considering the person is {age} years old)

(List features that score below 7/10)

---

### FEATURE-BY-FEATURE RATINGS

Provide a quick reference rating for ALL facial focal points:

| Feature | Rating | Notes |
|---------|--------|-------|
| Forehead Proportion | X/10 | Brief note |
| Eye Shape & Symmetry | X/10 | Brief note |
| Eye Spacing | X/10 | Brief note |
| Nose Shape | X/10 | Brief note |
| Nose Proportion | X/10 | Brief note |
| Lip Shape | X/10 | Brief note |
| Lip Ratio (upper:lower) | X/10 | Brief note |
| Cheekbone Definition | X/10 | Brief note |
| Jaw Definition | X/10 | Brief note |
| Chin Projection | X/10 | Brief note |
| Facial Symmetry | X/10 | Brief note |
| Overall Harmony | X/10 | Brief note |

---

### AGE-APPROPRIATE RECOMMENDATIONS

Given the person is {age} years old, here are realistic improvement options:

**Immediate/Non-Invasive:**
- Skincare focus areas
- Makeup/contouring techniques
- Facial exercises (mewing, etc.)

**Minimally Invasive (if desired):**
- Specific filler recommendations
- Botox considerations
- Other procedures appropriate for this age

**Long-term Considerations:**
- What to maintain
- What may change with age
- Preventative measures

---

### TOP 3 PRIORITY IMPROVEMENTS
Ranked by potential impact, considering age {age}:

1. **[Area]** - Why and how
2. **[Area]** - Why and how
3. **[Area]** - Why and how

---

Be encouraging and constructive. Remember that beauty is subjective and these are guidelines, not absolutes. Focus on enhancing natural features. Consider that at age {age}, certain recommendations may be more or less appropriate."""


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
        prompt = FACIAL_ANALYSIS_PROMPT.replace("{photo_description}", photo_description).replace("{age}", str(age))
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
        return jsonify({"error": str(e)}), 500


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
