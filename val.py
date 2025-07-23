import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
import csv

# ---- 配置 ----
DATA_DIR = r"D:\ai_detect\data\M4"  # 修改为你的数据目录
MODEL_PATH = r"D:\ai_detect\code\custom_roberta_text_classification_model\pytorch_model.bin"
MODEL_NAME = "roberta-base"
BATCH_SIZE = 8
MAX_LEN = 512
OUTPUT_CSV = "result.csv"
FAILED_FILE = "failed_files.txt"

# ---- Dataset 类 ----
class WikiDollyDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, mode='prompt_text'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt = item['prompt']
        text = item['text']

        # 保险：如果字段是 list，拼成 str
        if isinstance(prompt, list):
            prompt = " ".join(prompt)
        if isinstance(text, list):
            text = " ".join(text)

        if self.mode == 'prompt_text':
            text = prompt + ' ' + text
        elif self.mode == 'pure_text':
            text = text
        else:
            raise ValueError("Mode must be 'prompt_text' or 'pure_text'.")

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

# ---- 模型类 ----
class CustomRobertaModel(torch.nn.Module):
    def __init__(self, model_name):
        super(CustomRobertaModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits

# ---- 加载数据 ----
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            data.append({
                'prompt': obj['prompt'],
                'text': obj['human_text'],
                'label': 0
            })
            data.append({
                'prompt': obj['prompt'],
                'text': obj['machine_text'],
                'label': 1
            })
    return data

# ---- 验证函数 ----
def evaluate(model, data_loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    return recall, f1, acc, pre

# ---- 主批处理 ----
def main():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomRobertaModel(MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    # 第一次写结果文件表头
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Domain', 'Generator', 'Mode', 'Recall', 'F1', 'Acc', 'Pre'])

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.jsonl')]
    print(f"Found {len(files)} files in {DATA_DIR}")

    for file_name in files:
        base = file_name.replace('.jsonl', '')
        try:
            domain, generator = base.split('_', 1)
        except ValueError:
            domain = base
            generator = "Unknown"

        file_path = os.path.join(DATA_DIR, file_name)
        print(f"\nProcessing: {file_name}")

        try:
            data = load_jsonl(file_path)

            for mode in ['prompt_text', 'pure_text']:
                ds = WikiDollyDataset(data, tokenizer, max_len=MAX_LEN, mode=mode)
                loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

                recall, f1, acc, pre = evaluate(model, loader, device)
                print(f"{mode} => Recall: {recall:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}, Pre: {pre:.4f}")

                with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([domain, generator, mode, f"{recall:.4f}", f"{f1:.4f}", f"{acc:.4f}", f"{pre:.4f}"])

        except Exception as e:
            print(f"⚠️ Failed to process {file_name}: {e}")
            with open(FAILED_FILE, 'a', encoding='utf-8') as f:
                f.write(f"{file_name}\n")
            continue

    print(f"\nAll results saved to {OUTPUT_CSV}")
    print(f"Any failed files are recorded in {FAILED_FILE}")

if __name__ == "__main__":
    main()
