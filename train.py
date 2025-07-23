import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW
from sklearn.model_selection import train_test_split
import os

class TextClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        problem = str(self.data.iloc[index, 0])
        answer = str(self.data.iloc[index, 1])
        label = self.data.iloc[index, 2]

        problem_len = min(len(problem), self.max_len // 2)
        answer_len = self.max_len - problem_len

        encoding_problem = self.tokenizer.encode_plus(
            problem,
            add_special_tokens=True,
            max_length=problem_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        encoding_answer = self.tokenizer.encode_plus(
            answer,
            add_special_tokens=True,
            max_length=answer_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = torch.cat((encoding_problem['input_ids'], encoding_answer['input_ids']), dim=1)
        attention_mask = torch.cat((encoding_problem['attention_mask'], encoding_answer['attention_mask']), dim=1)

        return {
            'input_ids': input_ids.flatten(),
            'attention_mask': attention_mask.flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'problem_len': problem_len,
            'answer_len': answer_len
        }

class CustomRobertaModel(torch.nn.Module):
    def __init__(self, model_name, problem_attention_ratio=0.25, answer_attention_ratio=0.75):
        super(CustomRobertaModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.problem_attention_ratio = problem_attention_ratio
        self.answer_attention_ratio = answer_attention_ratio
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, problem_len, answer_len):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        last_hidden_state = outputs.last_hidden_state

        batch_size = last_hidden_state.size(0)
        representations = []

        for i in range(batch_size):
            pl = problem_len[i]
            al = answer_len[i]

            problem_representation = last_hidden_state[i, :pl, :].mean(dim=0)
            answer_representation = last_hidden_state[i, pl:pl+al, :].mean(dim=0)

            combined_representation = (
                self.problem_attention_ratio * problem_representation +
                self.answer_attention_ratio * answer_representation
            )
            representations.append(combined_representation)

        representations = torch.stack(representations)
        logits = self.classifier(representations)

        return logits

def custom_loss_function(outputs, labels, attention_mask, a):
    loss = torch.nn.CrossEntropyLoss()(outputs, labels)

    lengths = attention_mask.sum(dim=1).float()
    length_penalty = torch.mean(torch.abs(lengths - lengths.mean()))

    total_loss = loss + a * length_penalty

    return total_loss

def train_epoch(model, data_loader, optimizer, device, total_steps, current_step, accumulation_steps=4):
    model.train()
    total_loss = 0

    optimizer.zero_grad()

    for step, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        problem_len = batch["problem_len"]
        answer_len = batch["answer_len"]

        a = 0.00001 + 0.00009 * (1 - abs(2 * (current_step + step) / total_steps - 1))

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            problem_len=problem_len,
            answer_len=answer_len
        )

        loss = custom_loss_function(outputs, labels, attention_mask, a)
        loss = loss / accumulation_steps

        loss.backward()
        total_loss += loss.item()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (step + 1) % 200 == 0:
            print(f'Step {step + 1}, Loss: {loss.item() * accumulation_steps}, a: {a}')

    if (step + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(data_loader)


def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            problem_len = batch["problem_len"]
            answer_len = batch["answer_len"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                problem_len=problem_len,
                answer_len=answer_len
            )

            loss = custom_loss_function(outputs, labels, attention_mask, a=0)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader)

def save_model(model, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    torch.save(model.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))

    model_config = model.roberta.config
    model_config.save_pretrained(save_directory)

def load_model(save_directory):
    model = CustomRobertaModel('roberta-base', problem_attention_ratio=0.25, answer_attention_ratio=0.75)
    model.load_state_dict(torch.load(os.path.join(save_directory, 'pytorch_model.bin')))
    return model

if __name__ == '__main__':
    df = pd.read_csv('../data/processed_data_with_summary_unique_2.csv')

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    train_data = TextClassificationDataset(train_df, tokenizer, max_len=512)
    test_data = TextClassificationDataset(test_df, tokenizer, max_len=512)

    BATCH_SIZE = 32
    EPOCHS = 2
    LEARNING_RATE = 1e-5

    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = CustomRobertaModel('roberta-base', problem_attention_ratio=0.25, answer_attention_ratio=0.75)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps // 3, gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    current_step = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_loss = train_epoch(
            model,
            train_data_loader,
            optimizer,
            device,
            total_steps,
            current_step,
            accumulation_steps=3
        )

        current_step += len(train_data_loader)

        print(f'Train loss {train_loss}')

        test_acc, test_loss = eval_model(
            model,
            test_data_loader,
            device
        )
        print(f'Test accuracy {test_acc}')

    save_model(model, './model')

    tokenizer.save_pretrained('./model')
