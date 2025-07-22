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
        writer.writerow(["timestamp", "sensor_data", "llm_description", "confidence_score"])

# Preprocess incoming data to match model input format
def pre_process_data(data):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    # Ensure required protocol columns are present
    for col in ['protocol_TCP', 'protocol_UDP']:
        if col not in df.columns:
            df[col] = 0

    return df[['src_port', 'dst_port', 'packet_size', 'duration_ms', 'protocol_TCP', 'protocol_UDP']]

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

                # 🔸 Beautified terminal output
                print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print("📥  Incoming Sensor Data:")
                for k, v in data.items():
                    print(f"   • {k}: {v}")
                print(f"📉  Confidence Score: {score:.4f}")

                if prediction == -1:
                    print("\n🚨 Anomaly Detected! Sending to LLM...\n")

                    # Construct message for the LLM
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that labels and explains sensor anomalies."
                        },
                        {
                            "role": "user",
                            "content": f"Sensor reading: {json.dumps(data)}\nDescribe the type of anomaly and suggest a possible cause."
                        }
                    ]

                    # Send the message to Together AI LLM
                    try:
                        response = client.chat.completions.create(
                            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
                            messages=messages,
                            stream=False,
                        )
                        output = response.choices[0].message.content

                        print("🧠 LLM Response:")
                        print(f"{output}")
                        print("✅ Logged to anomalies.csv")
                        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

                        # Log the anomaly in CSV file
                        with open("anomalies.csv", "a", newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([datetime.now(), json.dumps(data), output, score])

                    except Exception as e:
                        print(f"❌ Error connecting to Together AI: {e}")
                        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

                else:
                    print("✔️  No anomaly detected.")
                    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

            except json.JSONDecodeError:
                print("❌ Error decoding JSON.")
