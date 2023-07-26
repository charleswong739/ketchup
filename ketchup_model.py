import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from cfg import MODEL_CHECKPOINT, SAVE_FILE, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH


class KetchupModel():

    def __init__(self):
        if gpu := torch.cuda.is_available():
            print(f'Running on GPU: {torch.cuda.get_device_name()}.')
            self.device = torch.device("cuda")
        else:
            print('Running on CPU.')
            self.device = torch.device("cpu")
        torch.manual_seed(0)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
        if gpu:
            self.model.to(self.device)

        self.model.load_state_dict(torch.load(SAVE_FILE, map_location='cuda' if gpu else 'cpu'))

    @torch.no_grad()
    def summarize(self, input: str):
        tokenized_input = self.tokenizer(input, max_length=MAX_INPUT_LENGTH,
                                    padding="max_length", truncation=True, return_tensors="pt")
        tokenized_input.to(self.device)

        outputs = self.model.generate(**tokenized_input, max_length=MAX_OUTPUT_LENGTH)

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return output_text
