import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import random
import torch
from torch.cuda.amp import autocast

# Clear existing cache
torch.cuda.empty_cache()

# Let's import the pipeline for LLM
# pipe = pipeline("text-generation", model="bigscience/bloom-7b1", torch_dtype=torch.float16)

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1", torch_dtype=torch.float16).to('cuda')

# # Enable data pallelism using torch
# model = torch.nn.DataParallel(bloom_llm)
# model = model.to('cuda')

# Evaluate mode
# model.eval()

# Bloom LLM
def ask_bloom_llm(input_text,
                  history,
                  tokenize: bool=True,
                  add_generation_prompt: bool=True):
    """
    This will take an input text, encode with the tokenizer,
    generate with the input_ids into the Bloom LLM, than decode
    the output id into text.
    """

    # # User's question
    # input_text = "How was jupiter created in the solar system."

    # Prompt template for LLM
    dialogue_template = [
        {"role": "user",
        "content": input_text}
    ]

    # Be sure the dialogue template is in string formate for the tokenizer
    prompt = ""
    for dialogue in dialogue_template:
        prompt += dialogue["content"] + " "
    
    # token id's for prompt
    input_ids = tokenizer(prompt, return_tensors='pt').to('cuda')

    # Bloom already comes in fp16

    # Let's use torch.no_grad() to save memory and computation
    with torch.no_grad():
        # Generate output from LLM
        outputs = model.generate(**input_ids,
                                max_new_tokens=256)

    # Decode the output tensors into string
    outputs_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return outputs_decoded

torch.cuda.empty_cache()

# Create the mushroom UI

chatbot=gr.Chatbot(height=700, label='Gradio ChatInterface')

with gr.Blocks(fill_height=True) as demo:
    gr.ChatInterface(
        fn=ask_bloom_llm,
        fill_height=True,
        title="Mushroom 🍄"
    )

if __name__ == "__main__":
    demo.launch()