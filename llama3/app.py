import gradio as gr
import os
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch

# Set an environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", None)


DESCRIPTION = '''
<div>
<h1 style="text-align: center;">Llama3 🦙</h1>
<p>This uses an open source Large Language Model called <a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B"><b>Llama3-8b</b></a></p>
</div>
'''

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16).to('cuda')
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

@spaces.GPU(duration=120)
def llama_generation(message: str, 
                     history: list, 
                     temperature: float, 
                     max_new_tokens: int
                     ) -> str:
    """
    Passes input, converts in tokens, generate's with ids and outputs
    the text out.
    """
    conversation = []
    for user, assistant in history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids= input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        eos_token_id=terminators,
    )
    # This will enforce greedy generation (do_sample=False) when the temperature is passed 0, avoiding the crash.             
    if temperature == 0:
        generate_kwargs['do_sample'] = False
        
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)
        

# Gradio block
chatbot=gr.Chatbot(height=600, label='Llama AI')

with gr.Blocks(fill_height=True) as demo:
    
    gr.Markdown(DESCRIPTION)
    gr.ChatInterface(
        fn=llama_generation,
        chatbot=chatbot,
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
        additional_inputs=[
            gr.Slider(minimum=0,
                      maximum=1, 
                      step=0.1,
                      value=0.95, 
                      label="Temperature", 
                      render=False),
            gr.Slider(minimum=128, 
                      maximum=4096,
                      step=1,
                      value=512, 
                      label="Max new tokens", 
                      render=False ),
            ],
        examples=[
            ["Make a poem of batman inside willy wonka"],
            ["How can you a burrito with just flour?"],
            ["How was saturn formed in 3 sentences"],
            ["How does the frontal lobe effect playing soccer"],
            ],
        cache_examples=False,
                     )
        
if __name__ == "__main__":
    demo.launch()
