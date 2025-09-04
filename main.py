def predict(csv_path, output_csv_path="predicted_emotions.csv"):
    import torch
    import pandas as pd
    import numpy as np
    from torch.utils.data import DataLoader, Dataset
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    from tqdm import tqdm

    model_dir = "emoModel"
    df = pd.read_csv(csv_path)

    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    texts = df["text"].tolist()

    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=128):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors="pt"
            )
            return {key: val.squeeze() for key, val in encoding.items()}

    test_dataset = TextDataset(texts, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_logits = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            all_logits.append(logits)

    all_logits = np.vstack(all_logits)
    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(int)
    preds_as_lists = preds.tolist()

    output_df = pd.DataFrame({
        "text": texts,
        "predicted_labels": preds_as_lists
    })

    output_df.to_csv(output_csv_path, index=False)
    print(f"Saved predictions to {output_csv_path}")

    return output_df
