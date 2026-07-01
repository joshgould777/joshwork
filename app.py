import json
import boto3
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

conversation_history = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    conversation_history.append({"role": "user", "content": user_message})

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": conversation_history,
        "system": "You are a helpful, friendly assistant. Be concise but thorough."
    })

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        body=body
    )

    response_body = json.loads(response["body"].read())
    assistant_message = response_body["content"][0]["text"]

    conversation_history.append({"role": "assistant", "content": assistant_message})

    return jsonify({"response": assistant_message})


@app.route("/clear", methods=["POST"])
def clear():
    global conversation_history
    conversation_history = []
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
