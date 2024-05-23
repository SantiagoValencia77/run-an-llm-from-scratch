import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tqdm
from sentence_transformers import SentenceTransformer, util
import re
from datetime import datetime, date
import time
from openai import OpenAI
import json
import os
from typing import Dict, Any, List
import textwrap
# from flask import Flask, request, jsonify
import gradio as gr

DESCRIPTION = '''
<div>
<h1 style="text-align: center;">Phobos 🪐</h1>
<p>This is a open tuned model that was fitted onto a RAG pipeline using <a href="https://huggingface.co/sentence-transformers/all-mpnet-base-v2"><b>all-mpnet-base-v2</b></a>.</p>
<h3 style="text-align: center;">In order to chat, please say 'gen phobos' = General Question you have of any topic. Say 'phobos' for questions specifically medical.</h3>
</div>
'''

# API keys
api_key = os.getenv('OPEN_AI_API_KEY')

df_embeds = pd.read_csv("chunks_tokenized.csv")
df_embeds["embeddings"] = df_embeds["embeddings"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

embeds_dict = df_embeds.to_dict(orient="records")

# convert into tensors
embeddings = torch.tensor(np.array(df_embeds["embeddings"].to_list()), dtype=torch.float32).to('cuda')


# Make a text wrapper
def text_wrapper(text):
    """
    Wraps the text that will pass here
    """

    clean_text = textwrap.fill(text, 80)

    print(clean_text)

# Let's first get the embedding model
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                      device='cuda')


# functionize RAG Pipeline

def rag_pipeline(query,
                 embedding_model,
                 embeddings,
                 device: str,
                 chunk_min_token: list):
    """
    Grabs a query and retrieve data all in passages, augments them, than it
    it outputs the top 5 relevant results regarding query's meaning using dot scores.
    """

    # Retrieval
    query_embeddings = embedding_model.encode(query, convert_to_tensor=True).to(device)

    # Augmentation
    dot_scores = util.dot_score(a=query_embeddings, b=embeddings)[0]

    # Output
    scores, indices = torch.topk(dot_scores, k=5)
    counting = 0
    for score, idx in zip(scores, indices):
        counting+=1
        clean_score = score.item()*100
        print(f"For the ({counting}) result has a score: {round(clean_score, 2)}%")
        print(f"On index: {idx}")
        print(f"Relevant Text:\n")
        print(f"{text_wrapper(chunk_min_token[idx]['sentence_chunk'])}\n")


# Message request to gpt
def message_request_to_model(input_text: str):
    """
    Message to pass to the request on API
    """
    message_to_model = [
        {"role": "system", "content": "You are a helpful assistant called 'Phobos'."},
        {"role": "user", "content": input_text}, # This must be in string format or else the request won't be successful
    ]

    return message_to_model


# Functionize API request from the very beginning as calling gpt for the first time
def request_gpt_model(input_text,
                      temperature,
                      message_to_model_api,
                      model: str="gpt-3.5-turbo"):
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
    print(output)
    return output, json_response

# Functionize saving output to file
def save_log_models_activity(query, prompt, continue_question, output, cont_output, embeds_dict, json_response,
                             model, rag_pipeline, message_request_to_model, indices, embedding_model, source_directed: str):
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
    
    clean_query = re.sub(r'[^\w\s]', '', input_query).replace(' ', '_')
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
    
        for idx in indices:
            # Let's log the models activity in txt file
            if rag_pipeline:
                file.write(f"{source_directed}")
                file.write(f"\n\nPipeline Used: RAG\n")
                file.write(f"Embedding Model used on tokenizing pipeline:\n\n{embedding_model}\n")
                
            file.write(f"\nRelevant Passages: {embeds_dict[idx]['sentence_chunk']}\n\n")
            break
        file.write(f"Model used: {model}\n")
        # file.write(f"{message_request_to_model}")
        today = date.today()
        current_time = datetime.now().time()
        file.write(f"Date: {today.strftime('%B %d, %Y')}\nTime: {current_time.strftime('%H:%M:%S')}\n\n")


# retrieve rag resources such as score and indices
def rag_resources(query: str,
                  device: str="cuda"):
    """
    Extracts only the scores and indices of the top 5 best results
    according to dot scores on query.
    """

    # Retrieval
    query_embeddings = embedding_model.encode(query, convert_to_tensor=True).to(device)

    # Augmentation
    dot_scores = util.dot_score(a=query_embeddings, b=embeddings)[0]

    # Output
    scores, indices = torch.topk(dot_scores, k=5)

    return scores, indices

# Format the prompt
def rag_prompt_formatter(prompt: str,
                         prev_quest: list,
                         context_items: List[Dict[str, Any]]):
    """
    Format the base prompt with the user query.
    """
    # Convert the list into string
    prev_questions_str ='\n'.join(prev_quest) # convert to string so we can later format on base_prompt

    context = "- " + "\n- ".join(i["sentence_chunk"] for i in context_items)

    base_prompt = """In this text, you will act as supportive medical assistant.
Give yourself room to think.
Explain each topic with facts and also suggestions based on the users needs.
Keep your answers thorough but practical.
\nHere are the past questions and answers you gave to the user, to serve you as a memory:
{previous_questions}
\nYou as the assistant will recieve context items for retrieving information.
\nNow use the following context items to answer the user query. Be advised if the user does not give you
any query that seems medical, DO NOT extract the relevant passages:
{context}
\nRelevant passages: Please extract the context items that helped you answer the user's question
<extract relevant passages from the context here>
User query: {query}
Answer:"""
    
    prompt = base_prompt.format(previous_questions=prev_questions_str, context=context, query=prompt)
    return prompt

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


# Function RAG-GPT
def rag_gpt(query: str,
            previous_quest: list,
            continue_question: str="",
            rag_pipeline: bool=True,
            temperature: int=0,
            model: str="gpt-3.5-turbo",
            embeds_dict=embeds_dict):
    """
    This contains the RAG system implemented with
    OpenAI models. This will process the the data through
    RAG, afterwards be formatted into instructive prompt to model
    filled with examples, context items and query. Afterwards,
    this prompt is passed the models endpoint on API and cleanly return's
    the output on response.
    """

    if continue_question == "":
        print(f"Your question: {query}\n")
    else:
        print(f"Your Question: {continue_question}\n")

    # Show query
    query_back = f"Your question: {query}\n"
    cont_query_back = f"Your Question: {continue_question}\n"
    top_score_back = ""
    # RAG resources
    # scores, indices = rag_resources(query)
    if rag_pipeline:
        scores, indices = rag_resources(query)
        # Get context item for prompt generation
        context_items = [embeds_dict[idx] for idx in indices]

        # augment the context items with the base prompt and user query
        prompt = rag_prompt_formatter(prompt=query, prev_quest=previous_quest, context_items=context_items)

        # Show analytics on response data   
        top_score = [score.item() for score in scores]
        print(f"Highest Result: {round(top_score[0], 2)*100}%\n")
        top_score_back += f"Highest Result: {round(top_score[0], 2)*100}%\n"

    else:
        prompt = general_prompt_formatter(prompt=query, prev_quest=previous_quest)
        print(f"Here is the previous 7 questions: {previous_quest}")
        print(f"This is the prompt: {prompt}")
        print(f"\nEnd of prompt")

    # all variables to return back to json on API endpoint for gardio
    cont_output_back = ""
    output_back = ""
    source_grabbed_back = ""
    url_source_back = ""
    pdf_source_back = ""
    link_or_pagnum_back = ""

    # LLM input prompt
    # If there is follow up question
    # Let's log the models activity in txt file    
    if continue_question != "":
        message_request = message_request_to_model(input_text=continue_question)
        cont_output, json_response = request_gpt_model(continue_question, temperature=temperature, message_to_model_api=message_request, model=model)
        cont_output_back += cont_output
        output = ""
        index = embeds_dict[indices[0]]
        # Let's get the link or page number of retrieval
        link_or_pagnum = index["link_or_page_number"]
        link_or_pagnum = str(link_or_pagnum)
        if link_or_pagnum.isdigit():
            link_or_pagnum_back += link_or_pagnum
            # link_or_pagnum = int(link_or_pagnum)
            source = f"The sources origins comes from a PDF"
            # source_back += source
            save_log_models_activity(query=query,
                                     prompt=prompt,
                                     continue_question=continue_question,
                                     output=output,
                                     cont_output=cont_output,
                                     embeds_dict=embeds_dict,
                                     json_response=json_response,
                                     model=model,
                                     rag_pipeline=rag_pipeline,
                                     message_request_to_model=continue_question,
                                     indices=indices,
                                     embedding_model=embedding_model,
                                     source_directed=source)

        else:
            link = f"Source Directed : {index['link_or_page_number']}"
            # link_back += link
            save_log_models_activity(query=query,
                                     prompt=prompt,
                                     continue_question=continue_question,
                                     output=output,
                                     cont_output=cont_output,
                                     embeds_dict=embeds_dict,
                                     json_response=json_response,
                                     model=model,
                                     rag_pipeline=rag_pipeline,
                                     message_request_to_model=continue_question,
                                     indices=indices,
                                     embedding_model=embedding_model,
                                     source_directed=link)
    
    # If no follow up question
    else:
        message_request = message_request_to_model(input_text=prompt)
        output, json_response = request_gpt_model(prompt, temperature=temperature, message_to_model_api=message_request, model=model)
        output_back += output
        cont_output = ""
        if rag_pipeline:
            index = embeds_dict[indices[0]]
            # Let's get the link or page number of retrieval
            link_or_pagnum = index["link_or_page_number"]
            link_or_pagnum = str(link_or_pagnum)
            if link_or_pagnum.isdigit():
                link_or_pagnum_back += link_or_pagnum
                print("is digit\n")
                source = f"The sources origins comes from a PDF"
                # source_back += source
                save_log_models_activity(query=query,
                                         prompt=prompt,
                                         continue_question=continue_question,
                                         output=output,
                                         cont_output=cont_output,
                                         embeds_dict=embeds_dict,
                                         json_response=json_response,
                                         model=model,
                                         rag_pipeline=rag_pipeline,
                                         message_request_to_model=query,
                                         indices=indices,
                                         embedding_model=embedding_model,
                                         source_directed=source)
    
            else:
                link = f"Source Directed : {index['link_or_page_number']}"
                # link_back += link
                save_log_models_activity(query=query,
                             prompt=prompt,
                             continue_question=continue_question,
                             output=output,
                             cont_output=cont_output,
                             embeds_dict=embeds_dict,
                             json_response=json_response,
                             model=model,
                             rag_pipeline=rag_pipeline,
                             message_request_to_model=query,
                             indices=indices,
                             embedding_model=embedding_model,
                             source_directed=link)
        else:
            save_log_models_activity(query=query,
                                     prompt=prompt,
                                     continue_question="",
                                     output=output,
                                     cont_output="",
                                     embeds_dict=embeds_dict,
                                     json_response=json_response,
                                     model=model,
                                     rag_pipeline=rag_pipeline,
                                     message_request_to_model="",
                                     indices="",
                                     embedding_model=embedding_model,
                                     source_directed="")
            
    if rag_pipeline:
        for idx in indices:
            print(f"\n\nOriginated Source:\n\n {embeds_dict[idx]['sentence_chunk']}\n")
            source_grabbed_back += f"\n\nOriginated Source:\n\n {embeds_dict[idx]['sentence_chunk']}\n"
            link_or_pagnum = embeds_dict[idx]['link_or_page_number']
            link_or_pagnum = str(link_or_pagnum)
            if link_or_pagnum.isdigit():
                link_or_pagnum = int(link_or_pagnum)
                print(f"The sources origins comes from a PDF")
                pdf_source_back += f"The sources origins comes from a PDF"
            else:
                print(f"Source Directed : {embeds_dict[idx]['link_or_page_number']}")  
                url_source_back += f"Source Directed : {embeds_dict[idx]['link_or_page_number']}"
            break

    else:
        pass

    if continue_question != "":
        return cont_output_back, source_grabbed_back, pdf_source_back, url_source_back

    else:
        return output_back, source_grabbed_back, pdf_source_back, url_source_back

# Mode of the LLM
llm_mode = "" 

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
    

def bot_comms(input, history):
    """
    Communication between UI on gradio to the rag_gpt model.
    """
    global llm_mode
    global memory_file_paths
    global prev_5_questions_list
    global first_time

    if input == "cuda info":
        output = check_cuda_and_gpu_type()
        return output
    
    state_mode = True
    # Input as 'gen phobos'
    if input == "gen phobos":
        output_text = "Great! Ask me any question. 🦧"
        llm_mode = input
        return output_text

    if input == "phobos":
        output_text = "Okay! What's your medical questions.⚕️"
        llm_mode = input
        return output_text
    
        # Reset memory with command
    if input == "reset memory":
        memory_file_paths = []
        output_text = f"Manually Resetted Memory! 🧠"
        return output_text

    if llm_mode == "gen phobos":
        # Get the 10 previous file paths
        for path in memory_file_paths:
            with open(path, 'r', encoding='utf-8') as file:
                q_a = file.read()
                # Now we have the q/a in string format
                q_a = str(q_a)
                # Make keys and values for prev dict
                prev_5_questions_list.append(q_a)

        if first_time:
            state_mode = False
            # Get the previous questions and answers list to pass to rag_gpt to place on base prompt
            gen_gpt_output = rag_gpt(input, previous_quest=[], rag_pipeline=state_mode)
            first_time = False
        else:
            state_mode = False
            gen_gpt_output = rag_gpt(input, previous_quest=prev_5_questions_list, rag_pipeline=state_mode)

        # reset the memory file_paths
        if len(memory_file_paths) == 5:
            memory_file_paths = []

        file_path = prev_recent_questions(input_text=input, ai_output=gen_gpt_output)
        memory_file_paths.append(file_path)
    
    if llm_mode == "phobos":
        for path in memory_file_paths:
            with open(path, 'r', encoding='utf-8') as file:
                q_a = file.read()
                # Now we have the q/a in string format
                q_a = str(q_a)
                # Make keys and values for prev dict
                prev_5_questions_list.append(q_a)

        if first_time:
            # Get the previous questions and answers list to pass to rag_gpt to place on base prompt
            rag_output_text = rag_gpt(input, previous_quest=[], rag_pipeline=state_mode)
            first_time = False
            # return jsonify({'output': rag_output_text})
        else:
            rag_output_text = rag_gpt(input, previous_quest=prev_5_questions_list, rag_pipeline=state_mode)
            # return jsonify({'output': rag_output_text})
        
        # reset the memory file_paths
        if len(memory_file_paths) == 5:
            memory_file_paths = []
        
        file_path = prev_recent_questions(input_text=input, ai_output=rag_output_text)
        memory_file_paths.append(file_path)

    output = rag_gpt(query=input,
                     previous_quest=[],
                     rag_pipeline=False)
    formatted_response = "\n".join(output[0].split("\n"))
    return formatted_response

# Gradio block
chatbot=gr.Chatbot(height=725, label='Gradio ChatInterface')

with gr.Blocks(fill_height=True) as demo:
    gr.Markdown(DESCRIPTION)
    gr.ChatInterface(
        fn=bot_comms,
        chatbot=chatbot,
        fill_height=True,
        examples=["gen phobos", "phobos", "reset memory", "cuda info"],
        cache_examples=False
    )

if __name__ == "__main__":
    demo.launch()