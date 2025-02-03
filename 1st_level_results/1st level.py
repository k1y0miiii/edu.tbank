import os
import json

import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_scheduler,
)


def set_huggingface_token(token: str) -> None:
    """
    Устанавливает токен и пути кеша для Hugging Face Hub в переменные окружения.

    Аргументы:
        token (str): Токен аутентификации для Hugging Face Hub
    """
    os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface/")
    os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser(
        "~/.cache/huggingface/transformers/"
    )
    os.environ["HF_DATASETS_CACHE"] = os.path.expanduser(
        "~/.cache/huggingface/datasets/"
    )
    os.environ["HF_METRICS_CACHE"] = os.path.expanduser(
        "~/.cache/huggingface/metrics/"
    )
    os.environ["HF_HUB_TOKEN"] = token
    print("Токен Hugging Face успешно установлен!")


set_huggingface_token("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")


def encode_examples(ds, tokenizer, limit: int = -1) -> TensorDataset:
    """
    Кодирует текстовые примеры в тензоры для модели.

    Аргументы:
        ds: Датасет с примерами
        tokenizer: Токенизатор для обработки текста
        limit (int, optional): Максимальное количество примеров. По умолчанию -1 (все)

    Возвращает:
        TensorDataset: Набор данных в виде тензоров
    """
    input_ids, attention_masks, labels = [], [], []
    
    if limit > 0:
        ds = ds.select(range(limit))
        
    for example in ds:
        text = example["prompt"] + " " + example["chosen"][1]["content"]
        encoded = tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids.append(encoded["input_ids"][0])
        attention_masks.append(encoded["attention_mask"][0])
        labels.append(float(example["chosen_rating"]))
        
    return TensorDataset(
        torch.stack(input_ids),
        torch.stack(attention_masks),
        torch.tensor(labels, dtype=torch.float),
    )


# Инициализация токенизатора и загрузка данных
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
dataset = load_dataset(
    "esfrankel17/HelpSteer2_binarized",
    split="average_rating_split"
)
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset, test_dataset = train_test_split["train"], train_test_split["test"]

# Подготовка DataLoader'ов
train_data = encode_examples(train_dataset, tokenizer)
val_data = encode_examples(test_dataset, tokenizer)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Инициализация модели
model = AutoModelForSequenceClassification.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    num_labels=1
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> dict:
    """
    Оценивает производительность модели на данных.

    Аргументы:
        model: Модель для оценки
        data_loader: Загрузчик данных
        device: Устройство для вычислений (CPU/GPU)

    Возвращает:
        dict: Результаты оценки модели
    """
    model.eval()
    metric = load_metric("accuracy")
    
    for batch in data_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=labels)
        
    return metric.compute()


pre_train_results = evaluate_model(model, val_loader, device)


def reinforce_with_baseline(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 1
) -> torch.nn.Module:
    """
    Обучает модель с использованием подхода REINFORCE с базовой линией.

    Аргументы:
        model: Модель для обучения
        train_loader: Загрузчик обучающих данных
        device: Устройство для вычислений (CPU/GPU)
        epochs (int, optional): Количество эпох. По умолчанию 1

    Возвращает:
        Обученную модель
    """
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_loader) * epochs
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    baseline = 0.0
    model.train()
    
    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            batch = [b.to(device) for b in batch]
            input_ids, attention_mask, labels = batch
            
            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            adjusted_loss = loss - baseline
            adjusted_loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            baseline = 0.9 * baseline + 0.1 * loss.item()
            
            if step % 10 == 0:
                print(
                    f"Эпоха: {epoch + 1}, Шаг: {step}, "
                    f"Потеря: {loss.item():.4f}, Базовый уровень: {baseline:.4f}"
                )
                
    return model


# Обучение модели
reinforced_model = reinforce_with_baseline(model, train_loader, device)

# Оценка после обучения
post_train_results = evaluate_model(reinforced_model, val_loader, device)

# Сохранение результатов
results = {
    "результаты_до_обучения": pre_train_results,
    "результаты_после_обучения": post_train_results,
}

print(json.dumps(results, indent=4, ensure_ascii=False))

# Сохранение модели и результатов
model.save_pretrained("./model_save_directory")
print("Модель сохранена в директории 'model_save_directory'.")

results_save_path = "validation_results.json"
with open(results_save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
    
print(f"Результаты валидации сохранены в файл: {results_save_path}")
