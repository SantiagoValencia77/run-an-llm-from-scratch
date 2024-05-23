import torch
import numpy as np
import pandas as pd
import tqdm
import re
from datetime import datetime, date
import time
from openai import OpenAI
import json
import os
from typing import Dict, Any, List
import textwrap
import gradio as gr
import whisper

DESCRIPTION = '''
<div>
<h1 style="text-align: center;">Saturn 🤖</h1>
<p>This is an open LLM model using GPT-4, Be free to ask it any questions!</p>
<h3 style="text-align: center;">If the model is being too slow, type 'reset memory' in order to clear the cache. 👍</h3>
<h4 style="text-align: center;">For more information regarding using this app: <a href="https://github.com/SantiagoValencia77/saturn/tree/main"><b>Click Here!</b></h4>
</div>
'''

# API keys
api_key = os.getenv('OPEN_AI_API_KEY')

# Message request to gpt
def message_request_to_model(input_text: str):
    """
    Message to pass to the request on API
    """
    message_to_model = [
        {"role": "system", "content": "You are a helpful assistant called 'Saturn'."},
        {"role": "user", "content": input_text}, # This must be in string format or else the request won't be successful
    ]

    return message_to_model


# Functionize API request from the very beginning as calling gpt for the first time
def request_gpt_model(input_text,
                      temperature,
                      message_to_model_api,
                      model: str="gpt-4o"):
    """
    This will pass in a request to the gpt api with the messages and
    will take the whole prompt generated as input as intructions to model
    and output the similiar meaning on the output.
    """
    # Create client
    client = OpenAI(api_key=api_key)

    # Make a request, for the input prompt
    response = client.chat.completions.create(
        model=model,
        messages=message_to_model_api,
        temperature=temperature,
    )

    # Output the message in readable format
    output = response.choices[0].message.content
    json_response = json.dumps(json.loads(response.model_dump_json()), indent=4)
    # print(f"{text_wrapper(output)}")
    # print(output)
    return output, json_response

# Functionize saving output to file
def save_log_models_activity(query, prompt, continue_question, output, cont_output, json_response, model):
    """
    This will save the models input and output interaction, onto
    a txt file, for each request, labeling model that was used.
    What sort of embedding process, pipeline that was used and
    date and time it was ran
    """
    # If there is a follow up question:
    input_query = ""
    if continue_question != "":
        input_query += continue_question
    else:
        input_query += query

    # Clean the query and limit max up to 20 characters for filename.
    clean_query = re.sub(r'[^\w\s]', '', input_query)[:20].replace(' ', '_')
    file_path = os.path.join("./logfiles/may-2024/", f"{clean_query}.txt")
    
    #Open the file in write mode
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f"Original Query: {query}\n\n")
        if prompt != "":
            file.write(f"Base Prompt: {prompt}\n\n")
        if continue_question != "":
            file.write(f"Follow up question:\n\n{continue_question}\n\n")
            file.write(f"Output:\n\n {cont_output}")
        else:
            file.write(f"Output:\n\n{output}\n\n")

        # Json response
        file.write(f"\n\nJson format response: {json_response}\n\n")
        file.write(f"Model used: {model}\n")
        # file.write(f"{message_request_to_model}")
        today = date.today()
        current_time = datetime.now().time()
        file.write(f"Date: {today.strftime('%B %d, %Y')}\nTime: {current_time.strftime('%H:%M:%S')}\n\n")


# Format general prompt for any question
def general_prompt_formatter(prompt: str,
                             prev_quest: list):
    """
    Formats the prompt to just past the 10 previous questions without
    rag.
    """
    # Convert the list into string
    prev_questions_str ='\n'.join(prev_quest) # convert to string so we can later format on base_prompt

    base_prompt = """In this text, you will act as supportive assistant.
Give yourself room to think.
Explain each topic with facts and also suggestions based on the users needs.
Keep your answers thorough but practical.
\nHere are the past questions and answers you gave to the user, to serve you as a memory:
{previous_questions}
\nAnswer the User query regardless if there was past questions or not.
\nUser query: {query}
Answer:"""
    prompt = base_prompt.format(previous_questions=prev_questions_str, query=prompt) # format method expect a string to subsistute not a list
    return prompt

# Saving 10 Previous questions and answers
def prev_recent_questions(input_text: str,
                          ai_output: list):
    """
    Saves the previous 10 questions asked by the user into
    a .txt file, stores those file in a list, when the len()
    of that list reaches 10 it will reset to expect the next 10
    questions and answer given by AI.
    """
    formatted_response = f"Current Question: {input_text}\n\n"

    # Convert the tuple elements to strings and concatenate them with the formatted_response
    formatted_response += "".join(str(elem) for elem in ai_output)

    # clean the query (input_text)
    clean_query = re.sub(r'[^\w\s]', '', input_text).replace(' ', '_')
    file_path = os.path.join("./memory/may-2024", f"{clean_query}.txt")

    # Let's save the content in the path for the .txt file
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(formatted_response)
            today = date.today()
            current_time = datetime.now().today()
            file.write(f"\n\nDate: {today.strftime('%B %d, %Y')}\nTime: {current_time.strftime('%H:%M:%S')}\n\n")
    except Exception as e:
        print(f"Error writing file: {e}")

    # # Make a list of the path names
    return file_path

# Handle audio uploads
def transcribe_audio(audio_file_path):
    """
    Transcribe audio input passed from gradio and outputs
    the audio frequency into number form.
    """
    model = whisper.load_model("base")
    result = model.transcribe(audio_file_path)
    return result['text']

# Function RAG-GPT
def gpt_4(query: str,
          previous_quest: list,
          continue_question: str="",
          temperature: int=0,
          model: str="gpt-4o"):
    """
    This contains the RAG system implemented with
    OpenAI models. This will process the the data through
    RAG, afterwards be formatted into instructive prompt to model
    filled with examples, context items and query. Afterwards,
    this prompt is passed the models endpoint on API and cleanly return's
    the output on response.
    """

    # if continue_question == "":
    #     print(f"Your question: {query}\n")
    # else:
    #     print(f"Your Question: {continue_question}\n")

    prompt = general_prompt_formatter(prompt=query, prev_quest=previous_quest)
    # print(f"Here is the previous 7 questions: {previous_quest}")
    # print(f"This is the prompt: {prompt}")
    # print(f"\nEnd of prompt")

    # all variables to return back to json on API endpoint for gardio
    cont_output_back = ""
    output_back = ""

    # LLM input prompt
    # If there is follow up question
    # Let's log the models activity in txt file    
    if continue_question != "":
        message_request = message_request_to_model(input_text=continue_question)
        cont_output, json_response = request_gpt_model(continue_question, temperature=temperature, message_to_model_api=message_request, model=model)
        cont_output_back += cont_output
        output = ""
        save_log_models_activity(query=query,
                                 prompt=prompt,
                                 continue_question=continue_question,
                                 output=output,
                                 cont_output=cont_output,
                                 json_response=json_response,
                                 model=model)
    
    # If no follow up question
    else:
        message_request = message_request_to_model(input_text=prompt)
        output, json_response = request_gpt_model(prompt, temperature=temperature, message_to_model_api=message_request, model=model)
        output_back += output
        cont_output = ""
        save_log_models_activity(query=query,
                                 prompt=prompt,
                                 continue_question=continue_question,
                                 output=output,
                                 cont_output=cont_output,
                                 json_response=json_response,
                                 model=model)

    if continue_question != "":
        return cont_output_back

    else:
        return output_back 

# List of files paths for memory
memory_file_paths = []

# first time condition
first_time = True

# Previous 5 questions stored in a dictionary for the memory of LLM
prev_5_questions_list = []
    
def check_cuda_and_gpu_type():
    # Your logic to check CUDA availability and GPU type
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)  # Get info about first GPU
        return f"CUDA is Available! GPU Info: {gpu_info}"
    else:
        return "CUDA is Not Available."


def bot_comms(input, history, audio_file=None):
    """
    Communication between UI on gradio to the rag_gpt model.
    """
    global llm_mode
    global memory_file_paths
    global prev_5_questions_list
    global first_time

    # Verify that if text was sent instead than remove the audio files said before stored in gradio cache (this won't remove the memory nor the log files)
    if input != "":
        audio_file = None

    if input == "cuda info":
        output = check_cuda_and_gpu_type()
        return output
        
        # Reset memory with command
    if input == "reset memory":
        memory_file_paths = []
        output_text = f"Manually Resetted Memory! 🧠"
        return output_text
    
    if audio_file is not None:
        audio_transcription = transcribe_audio(audio_file_path=audio_file)
        input = audio_transcription

    for path in memory_file_paths:
        with open(path, 'r', encoding='utf-8') as file:
            q_a = file.read()
            # Now we have the q/a in string format
            q_a = str(q_a)
            # Make keys and values for prev dict
            prev_5_questions_list.append(q_a)

    if first_time:
        # Get the previous questions and answers list to pass to gpt-4o to place on base prompt
        output = gpt_4(input, previous_quest=[])
        first_time = False
    else:
        output = gpt_4(input, previous_quest=prev_5_questions_list)

    # reset the memory file_paths
    if len(memory_file_paths) == 5:
        memory_file_paths = []

    # formatted_response = "\n".join(output[0].split("\n"))
    # return formatted_response
    return output

# Gradio block
chatbot=gr.Chatbot(height=525, label='Gradio ChatInterface')

with gr.Blocks(fill_height=True) as demo:
    gr.Markdown(DESCRIPTION)
    gr.ChatInterface(
        fn=bot_comms,
        chatbot=chatbot,
        fill_height=True,
        examples=[["reset memory", "cuda info"]],
        additional_inputs=[gr.Audio(type="filepath", label="Upload Audio")],
        cache_examples=False
    )

if __name__ == "__main__":
    demo.launch()