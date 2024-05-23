import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import numpy as np
import random
import torch
import re
# Clear existing cache
torch.cuda.empty_cache()


# Load model directly
tokenizer = AutoTokenizer.from_pretrained("Salesforce/xgen-7b-8k-inst", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Salesforce/xgen-7b-8k-inst", torch_dtype=torch.float16).to('cuda')

# Bloom LLM
def xgen(input_text,
         history):
    """
    This will take an input text, encode with the tokenizer,
    generate with the input_ids into the Bloom LLM, than decode
    the output id into text.
    """

    # # User's question
    # input_text = "How was jupiter created in the solar system."

    # Prompt template for LLM "context"
    header = (
        "A chat between a curious human and an artificial intelligence assistant called bubble bee. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
    )

    # token id's for prompt
    input_ids = tokenizer(header + input_text, return_tensors='pt').to('cuda')

    # Bloom already comes in fp16

    # Let's use torch.no_grad() to save memory and computation
    with torch.no_grad():
        # Generate output from LLM
        outputs = model.generate(**input_ids,
                                 max_new_tokens=256,
                                 top_k=100,
                                 eos_token_id=50256)

    # Decode the output tensors into string
    outputs_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # CLEAN UP TEXT
    output_text = outputs_decoded.replace(header, "").strip()
    output_text = re.sub(r'^Assistant:\s*', '', output_text)
    output_text = output_text.replace('<|endoftext\>', '').strip()

    return output_text

torch.cuda.empty_cache()

# Create the mushroom UI

chatbot=gr.Chatbot(height=700, label='Gradio ChatInterface')

with gr.Blocks(fill_height=True) as demo:
    gr.ChatInterface(
        fn=xgen,
        fill_height=True,
        title="Bubble Bee 🐝"
    )

if __name__ == "__main__":
    demo.launch()