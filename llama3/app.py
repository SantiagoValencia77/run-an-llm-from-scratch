import torch
import gradio as gr
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os

TOKEN = os.getenv('HF_AUTH_TOKEN')
login(token=TOKEN,
      add_to_git_credential=False)

# UI Description (HTML)
DESCRIPTION = '''
<div>
<h1 style="text-align: center;">Sirius 🐦‍⬛</h1>
<p>An Open LLM <b href="https://huggingface.co/meta-llama/Meta-Llama-3-8B">'Llama3 8b'</b>, Be sure to ask any questions!</p>
</div>
'''

# Model Instance
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B") # Placed on cpu
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", token=TOKEN, torch_dtype=torch.float16).to('cuda') # On GPU


# Create the model inputs and output function
def ask_llama3(input: str,
               history):
    """
    This will pass input into the tokenizer, those input_ids
    will pase into the model for generation and finally
    those output_ids will be decoded into text.
    """

    # Prompt template for LLM "context"
    header = (
        "A chat between a curious human and an artificial intelligence assistant called Sirius. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
    )

    input_ids = tokenizer.encode(header + input, return_tensors='pt').to('cuda')
    
    # Generate a response with the input_ids
    output_encoded = model.generate(input_ids=input_ids,
                                    max_new_tokens=256)
    
    # Decode the output
    output_decoded = tokenizer.decode(output_encoded[0], skip_special_tokens=True)

    return output_decoded

# Gradio UI
# def bot_comms(input_text,
#               history):
#     """
#     UI the communicates with gradio and the LLM.
#     """

# Gradio block
chatbot=gr.Chatbot(height=600, label="Llama Chatbot")

with gr.Blocks(fill_height=True) as demo:
    gr.Markdown(DESCRIPTION)
    gr.ChatInterface(
        fn=ask_llama3,
        chatbot=chatbot,
        fill_height=True,
        examples=['How do you make Spaghetti',
                  'How was the solar system formed',
                  'Where can learn to program',
                  'How was does air have hydrogen',
                  'Tell me a story of batman being the villian as blackbeard'],
        cache_examples=False
    )

if __name__ == "__main__":
    demo.launch()

