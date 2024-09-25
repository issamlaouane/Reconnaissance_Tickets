<<<<<<< HEAD
import re
import gradio as gr

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("debu-das/donut_receipt_v1.20")
model = VisionEncoderDecoderModel.from_pretrained("debu-das/donut_receipt_v1.20"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def process_document(image):
    # prepare encoder inputs
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    # prepare decoder inputs
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
          
    # generate answer
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    
    # postprocess
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    
    return processor.token2json(sequence)

description = "Démo Gradio pour Donut, une instance du modèle `VisionEncoderDecoderModel` affiné sur CORD (analyse de documents). Pour l'utiliser, il vous suffit de télécharger votre image et de cliquer sur `Submit`, ou de cliquer sur l'un des exemples pour les charger."
article = ""

demo = gr.Interface(
    fn=process_document,
    inputs="image",
    outputs="json",
    title="Reconnaissance des tickets de caisse 🧾",
    description=description,
    article=article,
    examples=[["example.jpg"], ["example_1.jpg"],["example_2.jpg"], ["example_3.jpg"],["example_4.jpg"]],
    cache_examples=False)

=======
import re
import gradio as gr

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("debu-das/donut_receipt_v1.20")
model = VisionEncoderDecoderModel.from_pretrained("debu-das/donut_receipt_v1.20"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def process_document(image):
    # prepare encoder inputs
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    # prepare decoder inputs
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
          
    # generate answer
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    
    # postprocess
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    
    return processor.token2json(sequence)

description = "Démo Gradio pour Donut, une instance du modèle `VisionEncoderDecoderModel` affiné sur CORD (analyse de documents). Pour l'utiliser, il vous suffit de télécharger votre image et de cliquer sur `Submit`, ou de cliquer sur l'un des exemples pour les charger."
article = ""

demo = gr.Interface(
    fn=process_document,
    inputs="image",
    outputs="json",
    title="Reconnaissance des tickets de caisse 🧾",
    description=description,
    article=article,
    examples=[["example.png"], ["example_1.png"],["example_2.png"], ["example_3.png"],["example_4.png"]],
    cache_examples=False)
demo.launch(share=True)