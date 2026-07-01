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

FACIAL_ANALYSIS_PROMPT = """You are an expert facial aesthetics analyst specializing in the golden ratio (phi = 1.618) and facial harmony. Analyze this face photo and provide detailed, actionable recommendations.

## Your Analysis Should Include:

### 1. FACIAL PROPORTIONS ASSESSMENT
Evaluate the face against ideal proportions:
- **Vertical Thirds**: Is the face evenly divided into thirds (hairline to brows, brows to nose base, nose base to chin)?
- **Horizontal Fifths**: Are the five vertical sections equal (outer face to outer eye, eye width, between eyes, eye width, outer eye to outer face)?
- **Golden Ratio Relationships**: Check key phi (1.618) relationships

### 2. FEATURE-BY-FEATURE ANALYSIS

**Forehead:**
- Proportion relative to face
- Shape and symmetry

**Eyes:**
- Width relative to face (should be 1/5 of face width)
- Distance between eyes (should equal one eye width)
- Shape and symmetry

**Nose:**
- Width (should equal eye width at alar base)
- Length relative to middle third
- Profile angle if visible

**Lips:**
- Upper to lower lip ratio (ideal is 1:1.6)
- Width relative to nose (should be 1.5x nose width)
- Symmetry

**Chin/Jaw:**
- Chin projection
- Jaw definition
- Lower third proportion

**Cheekbones:**
- Projection and definition
- Facial diamond shape

**Overall Symmetry:**
- Left vs right side comparison
- Notable asymmetries

### 3. PERSONALIZED RECOMMENDATIONS

Based on your analysis, provide specific recommendations in these categories:

**Non-Invasive Options:**
- Skincare focus areas
- Makeup/contouring techniques
- Facial exercises (if applicable)

**Minimally Invasive Options:**
- Specific filler placement recommendations
- Botox considerations
- Thread lift areas

**Surgical Options (if significant changes desired):**
- Procedures that could enhance harmony

### 4. PRIORITY RANKING
List the top 3-5 changes that would have the biggest impact on achieving facial harmony, ranked by impact.

### 5. STRENGTHS
Highlight the person's best features that already align well with aesthetic ideals.

---
Be encouraging and constructive. Remember that beauty is subjective and these are guidelines, not absolutes. Focus on enhancing natural features rather than achieving "perfection."

Provide your analysis in a clear, organized format."""


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
    """Analyze a face photo for golden ratio proportions and provide recommendations."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    if not allowed_image(image_file.filename):
        return jsonify({"error": f"Image type not allowed. Allowed: {', '.join(IMAGE_EXTENSIONS)}"}), 400

    try:
        # Read and encode image
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Determine media type
        extension = image_file.filename.rsplit('.', 1)[1].lower()
        media_type_map = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        media_type = media_type_map.get(extension, 'image/jpeg')

        # Call Claude with vision
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": FACIAL_ANALYSIS_PROMPT
                        }
                    ]
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
