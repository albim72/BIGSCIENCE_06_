import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import gradio as gr

# Sample data for demonstration
data = {
    "text": [
        "Chciałbym zamówić pizzę",
        "Pokaż mi pogodę w Warszawie",
        "Jakie są twoje godziny otwarcia?",
        "Czy mogę zmienić termin wizyty?",
        "Potrzebuję pomocy z logowaniem",
        "Zarezerwuj stolik na jutro",
        "Wyślij mi fakturę",
        "Jak mogę się z tobą skontaktować?",
        "Jaka jest cena tego produktu?"
    ],
    "intent": [
        "zamówienie_jedzenia",
        "sprawdzenie_pogody",
        "informacja_godziny",
        "zmiana_terminu",
        "wsparcie_logowanie",
        "rezerwacja",
        "żądanie_faktury",
        "kontakt",
        "informacja_cena"
    ]
}

df = pd.DataFrame(data)
df.to_csv("data/intents.csv", index=False)

# Encode labels
labels = df["intent"].unique()
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {v: k for k, v in label2id.items()}

# For demo, use a small pretrained model and manual mapping
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(labels)
)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Prediction function (mock for demo, you would fine-tune in real case)
def classify_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top = torch.argmax(probs, dim=1).item()
    return {id2label[top]: float(probs[0][top])}

# Gradio demo
demo = gr.Interface(fn=classify_intent,
                    inputs=gr.Textbox(lines=3, placeholder="Wpisz komunikat użytkownika..."),
                    outputs="label",
                    title="Klasyfikator Intencji",
                    description="Prosty model NLP do klasyfikacji intencji użytkownika.")

demo.launch()
