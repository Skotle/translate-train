"""
KoGPT2 TSA Q/A fine-tuning runner.

Input format:
    tsa_train/*.txt, tsa_valid/*.txt
    One line = question<TAB>answer

Training format:
    </s>질문: {question} 답변: {answer}</s>

Only answer tokens are used for loss; question/prompt tokens and padding tokens
are masked with -100.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import random
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup,
)


PROMPT_Q = "질문: "
PROMPT_A = " 답변: "


@dataclass
class TrainConfig:
    model_name: str = "skt/kogpt2-base-v2"
    train_dir: str = "./tsa_train"
    valid_dir: str = "./tsa_valid"
    output_dir: str = "./kogpt2_finetuned"
    max_len: int = 256
    batch_size: int = 8
    grad_accum: int = 4
    epochs: int = 5
    lr: float = 3e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    save_every: int = 1
    patience: int = 3
    seed: int = 42
    num_workers: int = 0
    log_every: int = 50


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Fine-tune KoGPT2 on TSA Q/A tab-separated data.")
    parser.add_argument("--model-name", default=TrainConfig.model_name)
    parser.add_argument("--train-dir", default=TrainConfig.train_dir)
    parser.add_argument("--valid-dir", default=TrainConfig.valid_dir)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--max-len", type=int, default=TrainConfig.max_len)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--grad-accum", type=int, default=TrainConfig.grad_accum)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--warmup-ratio", type=float, default=TrainConfig.warmup_ratio)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--save-every", type=int, default=TrainConfig.save_every)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    args = parser.parse_args()
    return TrainConfig(
        model_name=args.model_name,
        train_dir=args.train_dir,
        valid_dir=args.valid_dir,
        output_dir=args.output_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        save_every=args.save_every,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        log_every=args.log_every,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_qa_dir(directory: str) -> list[tuple[str, str]]:
    files = sorted(glob.glob(os.path.join(directory, "*.txt")))
    if not files:
        raise FileNotFoundError(f"'{directory}' 에 .txt 파일이 없습니다.")

    pairs: list[tuple[str, str]] = []
    for file_path in files:
        file_pairs: list[tuple[str, str]] = []
        with open(file_path, "r", encoding="utf-8") as file:
            for lineno, raw_line in enumerate(file, 1):
                line = raw_line.strip()
                if not line:
                    continue
                if "\t" not in line:
                    print(f"  [경고] {os.path.basename(file_path)} L{lineno}: 탭 없음, 건너뜀")
                    continue
                question, answer = line.split("\t", maxsplit=1)
                question = question.strip()
                answer = answer.strip()
                if question and answer:
                    file_pairs.append((question, answer))

        pairs.extend(file_pairs)
        print(f"  └ {os.path.basename(file_path)}: {len(file_pairs):,}쌍")

    print(f"  → 합계: {len(pairs):,}쌍\n")
    return pairs


class QADataset(Dataset):
    def __init__(
        self,
        pairs: Iterable[tuple[str, str]],
        tokenizer: PreTrainedTokenizerFast,
        max_len: int,
    ) -> None:
        self.examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        bos = tokenizer.bos_token or ""
        eos = tokenizer.eos_token or ""

        for question, answer in pairs:
            prompt = f"{bos}{PROMPT_Q}{question}{PROMPT_A}"
            full_text = f"{prompt}{answer}{eos}"
            encoded = tokenizer(
                full_text,
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)

            prompt_token_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            labels = input_ids.clone()
            labels[:prompt_token_len] = -100
            labels[attention_mask == 0] = -100

            self.examples.append((input_ids, attention_mask, labels))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.examples[index]


def build_tokenizer_and_model(config: TrainConfig, device: torch.device):
    print("[1] 모델 로드...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        config.model_name,
        bos_token="</s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    model = GPT2LMHeadModel.from_pretrained(config.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print(f"   파라미터: {sum(p.numel() for p in model.parameters()):,}")
    return tokenizer, model


def evaluate(model, valid_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, labels in valid_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            valid_loss += output.loss.item()
    return valid_loss / max(1, len(valid_loader))


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    print(f"[device] {device}")

    tokenizer, model = build_tokenizer_and_model(config, device)

    print("\n[2] 데이터 로드...")
    print(f"  [train: {config.train_dir}]")
    train_pairs = load_qa_dir(config.train_dir)
    print(f"  [valid: {config.valid_dir}]")
    valid_pairs = load_qa_dir(config.valid_dir)

    train_ds = QADataset(train_pairs, tokenizer, config.max_len)
    valid_ds = QADataset(valid_pairs, tokenizer, config.max_len)
    print(f"  → train {len(train_ds):,} / valid {len(valid_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    optimizer_steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum)
    total_steps = optimizer_steps_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    os.makedirs(config.output_dir, exist_ok=True)
    best_valid_loss = float("inf")
    patience_count = 0

    print(f"\n[3] 학습 시작 ({config.epochs}epoch / {total_steps} optimizer steps)")
    print("=" * 60)

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, (input_ids, attention_mask, labels) in enumerate(train_loader, 1):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss / config.grad_accum
            loss.backward()
            train_loss += output.loss.item()

            is_accum_step = step % config.grad_accum == 0
            is_last_step = step == len(train_loader)
            if is_accum_step or is_last_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % config.log_every == 0:
                print(
                    f"  epoch {epoch} | step {step}/{len(train_loader)} "
                    f"| loss {train_loss / step:.4f}"
                )

        avg_train = train_loss / max(1, len(train_loader))
        avg_valid = evaluate(model, valid_loader, device)
        ppl = math.exp(avg_valid) if avg_valid < 20 else float("inf")
        print(f"\n  ▶ epoch {epoch} | train={avg_train:.4f} | valid={avg_valid:.4f} | PPL={ppl:.2f}")

        if epoch % config.save_every == 0:
            save_path = os.path.join(config.output_dir, f"epoch{epoch}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  → 저장: {save_path}")

        if avg_valid < best_valid_loss:
            best_valid_loss = avg_valid
            patience_count = 0
            best_path = os.path.join(config.output_dir, "best")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"  → best 갱신 (valid_loss={best_valid_loss:.4f})")
        else:
            patience_count += 1
            print(f"  개선 없음 ({patience_count}/{config.patience})")
            if patience_count >= config.patience:
                print(f"  Early Stopping (epoch {epoch})")
                break

        print("-" * 60)

    print(f"\n[4] 학습 완료! best valid_loss={best_valid_loss:.4f}")


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
