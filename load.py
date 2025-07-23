import torch
from transformers import RobertaTokenizer, RobertaModel
import os

class CustomRobertaModel(torch.nn.Module):
    def __init__(self, model_name, question_ratio=0.25, answer_ratio=0.75):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.question_ratio = question_ratio
        self.answer_ratio = answer_ratio
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state
        half = self.roberta.config.max_position_embeddings // 2
        q_len = (attention_mask[:, :half] > 0).sum(dim=1).min().item()
        q_repr = last_hidden[:, :q_len, :].mean(dim=1)
        a_repr = last_hidden[:, q_len:, :].mean(dim=1)
        combined = self.question_ratio * q_repr + self.answer_ratio * a_repr
        return self.classifier(combined)

def load_model(model_dir):
    model = CustomRobertaModel('roberta-base')
    state_path = os.path.join(model_dir, 'pytorch_model.bin')
    model.load_state_dict(torch.load(state_path))
    return model

def classify_qa(model, tokenizer, device, max_length=512):
    model.question_ratio, model.answer_ratio = 0.75, 0.25
    question = input("Question: ")
    answer = input("Answer: ")
    q_enc = tokenizer.encode_plus(question, add_special_tokens=True, max_length=max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
    a_enc = tokenizer.encode_plus(answer, add_special_tokens=True, max_length=max_length,
                                  padding='max_length', truncation=True, return_tensors='pt')
    input_ids = torch.cat([q_enc['input_ids'], a_enc['input_ids']], dim=1).to(device)
    attention_mask = torch.cat([q_enc['attention_mask'], a_enc['attention_mask']], dim=1).to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
    print("AI" if logits.argmax(dim=1).item() == 1 else "Human")

def classify_text(model, tokenizer, device, max_length=512):
    model.question_ratio, model.answer_ratio = 0.0, 1.0
    text = input("Text: ")
    enc = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_length,
                                padding='max_length', truncation=True, return_tensors='pt')
    input_ids, attention_mask = enc['input_ids'].to(device), enc['attention_mask'].to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
    print("AI" if logits.argmax(dim=1).item() == 1 else "Human")

if __name__ == "__main__":
    model_dir = './model'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_dir).to(device)
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model.eval()

    mode = input("Select mode (QA/text): ").strip().lower()
    if mode == 'qa':
        classify_qa(model, tokenizer, device)
    elif mode == 'text':
        classify_text(model, tokenizer, device)
    else:
        print("Invalid mode. Enter 'QA' or 'text'.")
