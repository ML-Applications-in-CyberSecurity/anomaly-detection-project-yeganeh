import socket
import json
import pandas as pd
import joblib
from together import Together
import csv
import os
from datetime import datetime

# Server connection settings
HOST = 'localhost'
PORT = 9999

# Load the pre-trained anomaly detection model
model = joblib.load("anomaly_model.joblib")

# Initialize Together AI client with API key
client = Together(api_key="8e9b1c3c1f9cd2c2a5cc0960e2b4c170c3488da122f662ef776338147c961778")

# Create CSV file with headers if it doesn't exist
if not os.path.exists("anomalies.csv"):
    with open("anomalies.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "sensor_data",
            "confidence_score",
            "explanatory_output",
            "technical_output",
            "summary_output"
        ])

# Preprocess incoming data to match model input format
def pre_process_data(data):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    # Ensure protocol columns exist
    for col in ['protocol_TCP', 'protocol_UDP', 'protocol_UNKNOWN']:
        if col not in df.columns:
            df[col] = 0

    # Add engineered features
    df['packet_per_duration'] = df['packet_size'] / df['duration_ms']
    df['is_suspicious_src_port'] = df['src_port'].isin([1337, 9999, 6666]).astype(int)
    df['is_unknown_protocol'] = df.get('protocol_UNKNOWN', 0).astype(int)

    # Return features in model training order
    return df[[
        'src_port', 'dst_port', 'packet_size', 'duration_ms',
        'protocol_TCP', 'protocol_UDP', 'protocol_UNKNOWN',
        'packet_per_duration', 'is_suspicious_src_port', 'is_unknown_protocol'
    ]]


# Start connection to the socket server
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    buffer = ""
    print("✅ Client connected to server.\n")

    while True:
        chunk = s.recv(1024).decode()
        if not chunk:
            break
        buffer += chunk

        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            try:
                data = json.loads(line)
                processed = pre_process_data(data)
                prediction = model.predict(processed.values)[0]
                score = model.decision_function(processed.values)[0]

                print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print("📥  Incoming Sensor Data:")
                for k, v in data.items():
                    print(f"   • {k}: {v}")
                print(f"📉  Confidence Score: {score:.4f}")

                if prediction == -1:
                    print("\n🚨 Anomaly Detected! Sending to LLM...\n")

                    # Define different prompt styles
                    prompt_variants = {
                        "explanatory": [
                            {"role": "system", "content": "You are a helpful assistant that explains sensor anomalies."},
                            {"role": "user", "content": f"Sensor reading: {json.dumps(data)}\nExplain what makes this anomalous and why it might happen."}
                        ],
                        "technical": [
                            {"role": "system", "content": "You are a cybersecurity expert specialized in anomaly detection."},
                            {"role": "user", "content": f"Analyze this input: {json.dumps(data)}.\nExplain in technical terms why this might be considered anomalous."}
                        ],
                        "summary": [
                            {"role": "system", "content": "You are a concise summarizer of anomaly types."},
                            {"role": "user", "content": f"Quickly label the type of anomaly in this data: {json.dumps(data)}"}
                        ]
                    }

                    llm_outputs = {}

                    # Loop through each prompt type
                    for prompt_type, messages in prompt_variants.items():
                        try:
                            response = client.chat.completions.create(
                                model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
                                messages=messages,
                                stream=False,
                            )
                            output = response.choices[0].message.content
                            llm_outputs[prompt_type] = output

                            print(f"🤖 LLM Response [{prompt_type}]:\n{output}\n")

                        except Exception as e:
                            print(f"❌ Error for prompt '{prompt_type}': {e}")
                            llm_outputs[prompt_type] = "ERROR"

                    print("✅ All responses logged.\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

                    # Write to CSV
                    with open("anomalies.csv", "a", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.now(),
                            json.dumps(data),
                            score,
                            llm_outputs.get("explanatory", ""),
                            llm_outputs.get("technical", ""),
                            llm_outputs.get("summary", "")
                            
                        ])

                else:
                    print("✔️  No anomaly detected.")
                    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

            except json.JSONDecodeError:
                print("❌ Error decoding JSON.")
