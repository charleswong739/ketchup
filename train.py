from math import ceil

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_scheduler

from tqdm import trange, tqdm

from rouge_score import rouge_scorer

from cfg import MODEL_CHECKPOINT, SAVE_FILE, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH

import matplotlib.pyplot as plt

MODEL = None
TOKENIZER = None

#####################################
#           HYPERPARAMETERS         #
#####################################
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5


def preprocess_batch(data):
    tokenized_input = TOKENIZER(data["dialogue"], max_length = MAX_INPUT_LENGTH,
                                padding="max_length", truncation=True)

    targets = TOKENIZER(text_target=data["summary"], max_length = MAX_OUTPUT_LENGTH,
                        padding="max_length", truncation=True)

    tokenized_input["labels"] = targets["input_ids"]

    return tokenized_input


def preprocess_data(data):
    tokenized_data = data.map(preprocess_batch, batched=True)
    tokenized_data = tokenized_data.remove_columns("id")
    tokenized_data = tokenized_data.remove_columns("dialogue")
    tokenized_data = tokenized_data.remove_columns("summary")
    tokenized_data.set_format("torch")

    return tokenized_data


if __name__ == '__main__':
    if gpu := torch.cuda.is_available():
        print(f'Running on GPU: {torch.cuda.get_device_name()}.')
        device = torch.device("cuda")
    else:
        print('Running on CPU.')
        device = torch.device("cpu")
    torch.manual_seed(0)

    train_data = load_dataset("samsum", split="train")
    val_data = load_dataset("samsum", split="validation")

    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    if gpu:
        MODEL.to(device)

    train_dl = DataLoader(preprocess_data(train_data), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=gpu)
    val_dl = DataLoader(preprocess_data(val_data), batch_size=BATCH_SIZE, num_workers=2,
                        pin_memory=gpu)

    accumulation_steps = 8
    num_training_steps = NUM_EPOCHS * len(train_dl)
    num_optimizer_steps = ceil(num_training_steps / accumulation_steps)
    num_warmup_steps = num_optimizer_steps // 100
    num_winddown = num_optimizer_steps - num_warmup_steps
    optimizer = AdamW(MODEL.parameters(), lr=LEARNING_RATE)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer,
                                 num_warmup_steps=num_warmup_steps, num_training_steps=num_winddown)
    scaler = GradScaler()


    # DATA
    learning_rate = []
    validation_score = []
    loss_data = []


    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    best_rouge = 0
    MODEL.train()
    barfmt = ('{l_bar}{bar}| %d/' + str(NUM_EPOCHS)
              + ' [{elapsed}<{remaining}{postfix}]')
    with tqdm(total=num_training_steps, desc='Training', unit='epoch',
              bar_format=barfmt % 0, position=0, dynamic_ncols=True) as pbar:
        for epoch in trange(1, NUM_EPOCHS + 1):
            with tqdm(train_dl, desc=f'Epoch {epoch}', leave=False, unit='batch',
                      position=1, dynamic_ncols=True) as it:

                # train
                for i, batch in enumerate(train_dl):
                    batch = {k: v.to(device) for k, v in batch.items()}

                    with autocast():
                        outputs = MODEL(**batch)
                        loss = outputs.loss / accumulation_steps

                    learning_rate.append(lr_scheduler.get_last_lr())
                    loss_data.append(loss.item())

                    scaler.scale(loss).backward()
                    if ((i + 1) % accumulation_steps == 0) or (i + 1 == len(train_dl)):
                        scaler.step(optimizer)
                        scaler.update()

                        lr_scheduler.step()
                        optimizer.zero_grad()
                    it.set_postfix(loss=loss.item())
                    it.update()
                    pbar.update()

            # eval
            MODEL.eval()
            total_rouge_score = 0
            examples = 0
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    out = MODEL.generate(input_ids=batch["input_ids"], num_beams=2, max_length=MAX_OUTPUT_LENGTH)
                    decoded_out = TOKENIZER.batch_decode(out, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False)

                    decoded_labels = TOKENIZER.batch_decode(batch["labels"], skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False)

                    total_score = 0
                    for i in range(len(decoded_out)):
                        score = scorer.score(decoded_out[i], decoded_labels[i])

                        total_rouge_score += score['rougeL'].fmeasure
                        examples += 1

            mean_rouge_score = total_rouge_score / examples
            tqdm.write(f"Mean ROUGE fmesaure after epoch {epoch}: {mean_rouge_score}")
            validation_score.append(mean_rouge_score)

            if mean_rouge_score > best_rouge:
                torch.save(MODEL.state_dict(), SAVE_FILE)

            pbar.bar_format = barfmt % epoch


    # plot results
    fig, ((ax1, ax2), (ax3, _)) = plt.subplots(2, 2)
    save = False

    training_steps = [x for x in range(1, num_training_steps + 1)]

    ax1.plot(training_steps, learning_rate, "-")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Learning rate")

    ax2.plot(training_steps, loss_data, "-")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Loss")

    epochs = [x for x in range(1, NUM_EPOCHS + 1)]

    ax2.plot(training_steps, loss.cpu(), "-")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Score (Mean ROUGE-L F-measure)")

    plt.suptitle("Ketchup Model Training Metrics")
    plt.tight_layout()

    if save:
        plt.savefig("ketchup_training_metrics.png")

    plt.show()
