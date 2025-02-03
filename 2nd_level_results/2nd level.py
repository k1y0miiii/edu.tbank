import os
import json

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
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


class CustomModel(torch.nn.Module):
    """
    Кастомная модель для выдачи вероятностей с использованием предобученной основы.

    Аргументы:
        base_model_name (str): Название предобученной модели
        num_labels (int, optional): Количество классов. По умолчанию 10
    """

    def __init__(self, base_model_name: str, num_labels: int = 10) -> None:
        super().__init__()
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Прямой проход с вычислением вероятностей."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        return torch.softmax(logits, dim=1)


def prepare_labels(labels: list, num_labels: int = 10) -> np.ndarray:
    """
    Преобразует метки в one-hot encoding.

    Аргументы:
        labels (list): Список меток
        num_labels (int, optional): Количество классов. По умолчанию 10

    Возвращает:
        np.ndarray: One-hot представление меток
    """
    return np.eye(num_labels)[(np.array(labels) - 1).astype(int)]


def encode_examples(
    ds,
    tokenizer: AutoTokenizer,
    num_labels: int = 10
) -> TensorDataset:
    """
    Кодирует примеры в тензоры для обучения.

    Аргументы:
        ds: Исходный датасет
        tokenizer: Токенизатор для обработки текста
        num_labels (int, optional): Количество классов. По умолчанию 10

    Возвращает:
        TensorDataset: Набор данных в виде тензоров
    """
    input_ids, attention_masks, labels = [], [], []
    
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
        labels.append(prepare_labels([example["chosen_rating"]], num_labels))
        
    return TensorDataset(
        torch.stack(input_ids),
        torch.stack(attention_masks),
        torch.tensor(labels, dtype=torch.float32),
    )


# Инициализация компонентов
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
dataset = load_dataset(
    "esfrankel17/HelpSteer2_binarized",
    split="average_rating_split"
)
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Подготовка данных
train_data = encode_examples(train_dataset, tokenizer, 10)
val_data = encode_examples(test_dataset, tokenizer, 10)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Инициализация модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomModel("HuggingFaceTB/SmolLM2-135M-Instruct", 10)
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader),
)


def train_reinforce(
    model: CustomModel,
    train_loader: DataLoader,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> tuple:
    """
    Обучает модель с использованием метода REINFORCE.

    Аргументы:
        model: Модель для обучения
        train_loader: Загрузчик обучающих данных
        optimizer: Оптимизатор
        scheduler: Планировщик скорости обучения
        device: Устройство для вычислений

    Возвращает:
        tuple: Обученная модель и список потерь
    """
    model.train()
    losses = []
    
    for epoch in range(1):
        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            model.zero_grad()
            probabilities = model(input_ids, attention_mask)
            loss = -torch.sum(labels * torch.log(probabilities)) / labels.shape[0]
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            
            if step % 10 == 0:
                print(
                    f"Эпоха: {epoch + 1}, Шаг: {step}, "
                    f"Потеря: {loss.item():.4f}"
                )
                
    return model, losses


# Обучение модели
reinforced_model, training_losses = train_reinforce(
    model, train_loader, optimizer, lr_scheduler, device
)
print("Обучение методом REINFORCE завершено.")

# Сохранение результатов
results = {"потери_при_обучении": training_losses}
print(json.dumps(results, indent=4, ensure_ascii=False))

results_save_path = "validation_results.json"
with open(results_save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
    
print(f"Результаты валидации сохранены в файл: {results_save_path}")
