### RAG with GPT-3.5-Turbo

In this workspace, I used GPT-3.5-turbo as the llm and still contained the same rag pipeline i had with gemma.

This project is really no different from the gemma one, you need to create the pipeline process, than instance that pipeline along with the LLM (Place tokenizers)

Now this is a bit different as your using more of an API for the LLM and configure it a bit differently compared to `gemma` but that's good.

### You need to do this project on your own.

You can view the app.py that I made but only as a reference or cheatsheet when your really not sure, use it sort of like a review of the way how I did it but i want you to do it your own way, and i want you to remember and already have in your head how to create, chunk, batch, tokenize in text generation. You can  re-use the existing chunks_tokinzed.csv you made in gemma, since your using the same rag but if you want to make the rag all over again but with your data and in a different subject such as sports: soccer, basketball or in video games or anything else, your more than welcome too and I actually recommend that 😄. I just did this in a regular .py file, which is best. But since your beggining and you need reminders to yourself in markdown and also run just a few cells to make sure what your doing is right, than you can do that. No worries.


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
export OPEN_AI_API_KEY=place_your_secret_key_here
~~~
this just click on 'ctrl + o'(Write out) and 'Enter'(Save) than 'ctrl + x'(close), you don't need to '' quote your value in the api key.
To commit the changes to your console just type:
~~~
source ~/.bashrc
~~~

### Hugging Face Spaces

In hugging face, since your containerizing your project to use the environments you need while also connecting to hugging face hardware, you need to place your api key in the secret variables spaces has, you can find this if you go to settings in your spaces workspace.

Just simply place your variable name on the name and value on value.


### Google Colab

In colab there are 2 ways to do this

**Storing directly in colab**:
In the left section you will see a key symbol in the side menu bar, click on it. From here you can click on add new secret and you just enter the name and value.
However. There is a issue with this, this method is very practical but if you decide to share your notebook with anyone else that has edit access to the notebook, they will see the secret key and it's value, so PLEASE use this **WITH CAUTION**.

**Storing in Secret Manager inside google cloud**
Google Cloud offers you a way to store your secret environment variable with a lot of security so that no one can access besides you, here is the way to do it: https://g.co/gemini/share/7cae88c83805

The onle downside to this is that even though you can have 10k request for free, afterwards you will get billed. Even if your using it for free now, you need to place you billing info as this 'Free' service it actually more a trial google gives to everyone who is new.

In this repository I have the environment variable set as 'OPEN_AI_API_KEY' but you can change it to whatever name you put yours 



### We used Gradio as a UI between the User and AI

Compared to gemma we didn't add an application where a user can easily write a question and the LLM simply responds, well now we are. 
View the gradio docs for more information on the gradio app: https://www.gradio.app/guides/creating-a-chatbot-fast

The rest of the models I will show you here are the ones I have established in my Hugging Face Spaces: https://huggingface.co/sandz7

This one was derived from *Chat with phobos*, if you want to learn how to create your own space and configure a repo for your locally to communicate with your space.
Use this: https://huggingface.co/docs/hub/en/repositories-getting-started
