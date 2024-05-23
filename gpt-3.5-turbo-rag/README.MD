### RAG with GPT-3.5-Turbo

In this workspace, I used GPT-3.5-turbo as the llm and still contained the same rag pipeline i had with gemma.

This project is really no different from the gemma one, you need to create the pipeline process, than instance that pipeline along with the LLM (Place tokenizers)

Now this is a bit different as your using more of an API for the LLM and configure it a bit differently compared to `gemma` but that's good.

### You need to do this project on your own.

You can view the app.py that I made but only as a reference, use it sort of like a review of the way how i did it but i want you to remember and already have in your head how to create, chunk, batch, tokenize and generate in text generation. You can already re-use the existing chunks_tokinzed.csv you made in gemma, since your using the same rag but if you want to make the rag all over again but let's now with your data and in a different subject such as sports: soccer, basketball or in video games or anything else, your more than welcome too and I actually recommend that 😄


## OpenAI Configuration
I used my API keys to get access to the model, in order for you to use openai's gpt, you need to pay a credit for it, to be honest it's very cheap, i added $5 to my account and I've used it more than 100 times using both gpt-3.5 and gpt4o and i still have more than $4.25.

Here is the API reference you should use to get started for it: https://platform.openai.com/docs/api-reference/introduction
This is another really good docs to instance the gpt: https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models

You are going to need an account with openai, so make one. I think by default when you do, it already makes a developer account for you too in any case when you get to this page: https://platform.openai.com/apps, go to API and click it.

From there just login in the top right corner, once you logged in you should proceed to make individual or place your company name in the default project, by start openai will make a default project for you and you can only have your model in 1 project which is the default one. After placing your orginization and details, go to 'API Keys' in the left side bar menu, in there click 'Create new secret key' and name whatever you want. Most importantly copy the secret key value and store it somewhere you can use it because your going to have to place it in several places.

#### Windows
Of course if your running this locally you need to store in your Environment Variables, this can depend on the OS your using, if your on windows, just go to 'Control Panel' -> 'System and Security' -> 'System' -> 'Advanced system settings' -> 'Environment Variables', in this window look at 'Use variables for ['User'], in this box click on 'New' and place the environment variable name and the value you took from the secret key on openai, than click on save and click on 'OK' and 'OK' again.

#### Linux or WSL
Go to Ubuntu terminal, than type:
~~~
nano ~/.bashrc
~~~
This bashrc file, acts as sort of the same as storing your local environment and configuration but in this case it's in a script.
So, in the bottom you can place:
~~~
# Openai API keys
export OPEN_AI_API_KEY=['place_your_secret_key_here']
~~~
this just click on 'ctrl + o'(Write out) and 'Enter'(Save) than 'ctrl + x'(close)
To commit the changes to your console just type:
~~~
source ~/.bashrc
~~~

In this repository I have the environment variable set as 'OPEN_AI_API_KEY' but you can change it to whatever name you put yours 



### We used Gradio as a UI between the User and AI

Compared to gemma we didn't add an application where a user can easily write a question and the LLM simply responds, well now we are. 
View the gradio docs for more information on the gradio app: https://www.gradio.app/guides/creating-a-chatbot-fast

The rest of the models I will show you here are the ones I have established in my Hugging Face Spaces: https://huggingface.co/sandz7

This one was derived from *Chat with phobos*, if you want to learn how to create your own space and configure a repo for your locally to communicate with your space.
Use this: https://huggingface.co/docs/hub/en/repositories-getting-started