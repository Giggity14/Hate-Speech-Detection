import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import re

# ------------------------
# Text cleaning function
# ------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)              # remove mentions
    text = re.sub(r"#", "", text)                 # remove hashtags
    text = re.sub(r"[^\x00-\x7F]+", " ", text)   # keep ASCII (optional: remove non-ASCII)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------------
# Load model and tokenizer
# ------------------------
MODEL_DIR = "hsd/saved_model"
  # must point to your local folder
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ------------------------
# Label mapping
# ------------------------
id2label = {0: "Non-Hate Speech", 1: "Hate Speech"}

# ------------------------
# Classification function
# ------------------------
def classify_text(raw_text):
    text = clean_text(raw_text)
    
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        label_idx = int(probs.argmax())
        confidence = float(probs[label_idx])
        label = id2label.get(label_idx, str(label_idx))

    return {"label": label, "confidence": confidence, "probs": probs.tolist()}

# ------------------------
# Interactive testing
# ------------------------
if __name__ == "__main__":
    print("=== Hate Speech Detection Test ===")
    while True:
        text = input("\nEnter text (or 'exit' to quit): ")
        if text.lower() == "exit":
            break
        result = classify_text(text)
        print(f"Prediction: {result['label']} | Confidence: {result['confidence']:.4f}")
