import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from cfg import MODEL_CHECKPOINT, SAVE_FILE, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH

if __name__ == '__main__':
    if gpu := torch.cuda.is_available():
        print(f'Running on GPU: {torch.cuda.get_device_name()}.')
        device = torch.device("cuda")
    else:
        print('Running on CPU.')
        device = torch.device("cpu")
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    if gpu:
        model.to(device)

    model.load_state_dict(torch.load(SAVE_FILE, map_location='cuda' if gpu else 'cpu'))

    # input_text = "Isabella: fuck my life, I'm so not able to get up to work today\r\n" \
    #              "Isabella: I need to call in sick :(\r\n" \
    #              "Oscar: haha, well you certainly had a good time at the Christmas party yesterday XD\r\n" \
    #              "Isabella: shut up, you're a traitor\r\n" \
    #              "Isabella: I told you to guard my glass\r\n" \
    #              "Isabella: and my sobriety. You clearly failed!\r\n" \
    #              "Oscar: but you were having such fun, I didn't have a heart to stop it\r\n" \
    #              "Oscar: <file_photo>\r\n" \
    #              "Oscar: <file_photo>\r\n" \
    #              "Isabella: you're so dead! Is that Jimmy from marketing department?\r\n" \
    #              "Oscar: yes indeed, it's him :D\r\n" \
    #              "Isabella: I am a fallen woman, I cannot get back to the office now\r\n" \
    #              "Isabella: <file_gif>\r\n" \
    #              "Oscar: oh come on, almost everybody was drunk\r\n" \
    #              "Oscar: so they won't remember a thing :D\r\n" \
    #              "Isabella: I assure you, they tend to remember such things…\r\n" \
    #              "Oscar: <file_gif>"
    input_text = "Charlezz: Y’all wanna play catan\n" \
                 "Charlezz: I’m bored\n" \
                 "Moist Mortles: Unfortunately, I've filled my fun quota for the weekend\n" \
                 "Moist Mortles: it's schoolwork time for me\n" \
                 "Charlezz: L tbh\n" \
                 "Charlezz: jkjk we support scholarly activities\n" \
                 "Moist Mortles: thank you, mr. wong, your support in these times of tribulation and uncertainty is invaluable\n"

    tokenized_input = tokenizer(input_text, max_length = MAX_INPUT_LENGTH,
                                padding="max_length", truncation=True, return_tensors="pt")
    tokenized_input.to(device)

    outputs = model.generate(**tokenized_input, max_length=MAX_OUTPUT_LENGTH)

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    print(f"Transcript={input_text}\n--------------------------------\n")
    print(f"Summary={output_text}")


