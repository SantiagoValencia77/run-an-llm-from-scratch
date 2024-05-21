## Run An Gen-LLM From Scratch
Today I'll show you how to run several open source Generative LLM using Python, We'll look at a view concepts, than just pure code. If you want to do the same thing did, you can feel free to follow the steps I'll show 😊.

Let's start!


There are several ways you can run this, on jupyter notebook or on visual studio code, it's really up too you. I'm just going to stick to VS Code (Visual Studio Code) as tradition. A really good advice I can give for deep learning workspaces is to stick close to the Linux environment because nvidia configuration works really smoothly on Linux compared to windows, so to avoid hassle in the future, I recommend to you that if you already have Linux as your OS, well your setup. If you have windows, no worries, we can install WSL 2 on your OS to work as a bridge to Linux distribution such as Ubuntu.


If you are using windows follow the steps i'll show. Otherwise just ignore this and skip too (predefined skip):

> To install WSL2, follow this link instructions: https://learn.microsoft.com/en-us/windows/wsl/install
I can't tell were in the future you will see this so the steps may change which is why the best way is to look at the document from Microsoft themselves.
> 
After you have downloaded WSL2, you can make sure that you have (This can be on something like Powershell):
~~~
wsl --install --verbose
~~~
You should get something like this:
~~~
  NAME                   STATE           VERSION
* Ubuntu                 Running         2
  docker-desktop-data    Stopped         2
  docker-desktop         Stopped         2
~~~
Ignore the docker, that shows up if you don't have docker desktop. What's important though is to see that distribution name enabled such as 'Ubuntu', check the version number as well, I'd say make sure it's version 2.

If you ran into any sort of issues here, let's say that once you install wsl, the terminal prompted: That you need you enabled a VM on your BIOS, I'd say talk to the manufactures of your computer and try to resolve it with them, In the meantime though do not worry, while you get a response you can look at this github repo, on doing this project with windows, so that you don't get left behind: https://github.com/SantiagoValencia77/gen-llm-on-windows

Now that you have Ubuntu, now we to to configure a few things with nvidia.. Since Deep Learning often uses computational levels that GPU process faster and easier compared to CPU, you are going to have use a GPU. If you have an nvidia gpu that has a vram > = 11GB, than your good. If not, no worries, just use google collab for now.

### Nvidia Installation

If you're a macbook user, ignore this step, everyone else view this:

1. Install Nvidia CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit I advice to install it in a folder around your root directory, because you will need to touch these folders often, and online it's usually just referred to this area
2. Install Nvidia cuDNN Library: https://developer.nvidia.com/cudnn-downloads

Once you installed these Badboys in the right directory, that's easy to reach. Now we get to start working on the project.


Let's go to Google Drive (We need this for google colab). In their make a folder called 'deep-learning', than make a subfolder called 'llms'.

Now head to Google Colab, In there by default it will show a screen to open or create a notebook, in our case we created a folder to store the notebook, so head to that folder, than create the notebook there.

![image](https://github.com/SantiagoValencia77/run-an-llm-from-scratch/assets/159969500/3a1691b8-f08e-4224-a55c-686c23364b7e)

Head to 'File' than 'New Notebook'

![image](https://github.com/SantiagoValencia77/run-an-llm-from-scratch/assets/159969500/0f4cee45-b1c3-49db-80e3-58b3c909e5aa)

Click on the name and change it to whatever you want

Now to keep things clean. Click on file again and find 'move', click on it. In this screen, try to navigate into the folder you made in drive. (google colab connects it's data into the files in drive).

Now by default, colab keeps the most common dependencies inside there environments for Machine Learning and Deep Learning, so when you make a notebook you already have these environments. However, you don't have all of them, and there are several dependencies we will use that aren't installed into colab. To go across this, we're just going to grab and place things into the drive itself.


Make a file really anywhere on your computer called requirements.txt, once you open this file, you should have this content inside it:
~~~
torch
PyMuPDF==1.23.26
matplotlib
numpy
pandas
Requests==2.31.0
sentence_transformers==2.5.1
spacy
tqdm==4.66.2
transformers==4.38.2
accelerate
bitsandbytes
jupyter
wheel
lxml_html_clean
newspaper3k
flash-attn
~~~

Save the file, now head back into google drive where you made the folder for llm (we're the notebook is in) and save the requirements.txt file you just made in there.

![image](https://github.com/SantiagoValencia77/run-an-llm-from-scratch/assets/159969500/e5ab9634-86ed-44c9-b7fb-1a4a670d7311)

Placed requirements.txt in the same folder.

Next, on the notebook mount your drive onto the notebook so you can easily access your files from the there, after writing the code below, make sure to click on "shift + enter" to run that code cell:
~~~
from google.colab import drive, files
drive.mount('/content/drive')
~~~

Sign in with your google account (One your using in google drive)

![image](https://github.com/SantiagoValencia77/run-an-llm-from-scratch/assets/159969500/74bce54f-1147-4d1c-836b-ff26e778166b)

You should get a return prompt like this:
~~~
Mounted at /content/drive
~~~
Once you mounted the drive, click on the folder button on the left side bar:

![image](https://github.com/SantiagoValencia77/run-an-llm-from-scratch/assets/159969500/589dba82-bea9-4755-a6b9-5998914c6398)

Click on drive

![image](https://github.com/SantiagoValencia77/run-an-llm-from-scratch/assets/159969500/36d37703-0f9d-427a-811e-2fcd337ef9bb)

Find the requirements.txt you uploaded into drive, afterwards right-click on the file and select 'copy path'.

Before we upload the requirements.txt.. We need to connect a GPU that google offers us, we can do this by click on the top right arrow and selecting 'change runtime type':

![image](https://github.com/SantiagoValencia77/run-an-llm-from-scratch/assets/159969500/bdf10047-bc5c-4c6f-9483-6054da2dad70)


Now you should see under "Hardware Accelerator" the different hardware google gives to connect to the notebook, in our case we aren't going to use the standard CPU because that would be to slow for the instance of the model, so let's choose the free T4 GPU, it will likely tell you that it will terminate the session but that doesn't matter, just rerun all the cells we did:

![image](https://github.com/SantiagoValencia77/run-an-llm-from-scratch/assets/159969500/a0fe051d-ce27-4b84-8411-ad5db2a46bdc)

Click on save

One thing you should know about google colab is that there are time limits when you run hardware, there isn't an exact time limit they give you because that information isn't public but due to my experience, I found that google colab usually time's out your session when you've been using it for around 3–4 hours straight, afterwards I really found that you can't use any hardware until 24 hours pass.. I know this isn't ideal but it's the best option that's available right now, especially for free 🤷‍♂️

Now to make sure your T4 GPU is connected, you can run this code, which will check if the GPU is running and list the name of the GPU, otherwise if not it will print that there isn't a GPU:
~~~
import torch
# This will show a bool response (True/False)
if torch.cuda.is_available():
  # Get the first GPU name if condition is True
  # This 1st GPU is fine because we're only using 1 GPU
  print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
  # Otherwise if False, print this..
  print("GPU not available")
~~~
If for some reason you see that the GPU wasn't available, please be sure you wrote this exact code I did, otherwise contact google colab, contacting them can be cumbersome, so I just recommend to go to the repo your looking on my github and leave a comment, so me and the community can see if we can help you! As always get help from searching on google from the problem your facing, that's the best tool to use as a developer and also ask ChatGPT if it's too confusing, it can guide you with your specific setup, Look online, than if you can't find a solution say on Stack Overflow, than just comment on the repo to see if we can help you get an exact answer :)


After seeing a response like this: "GPU: Tesla T4" we can now go to the next step. On the cell below *(FYI: If your not inside a cell, clicking 'a' will create a cell above and clicking 'b' will make a cell below, to delete cells it's 'x')* in the notebook install the requirements.txt from that path your copied:
~~~
# Be sure to that you place on the path here instead, the path you copied earlier from requirements.txt
!pip install -r "/content/drive/MyDrive/deep-learning/demonstration/llms/requirements.txt"
~~~
When you run this at the end it will tell you that you have to Restart your session which is fine, do that and you will need to re run all the cells again but the dependencies you just installed should be in your environment now. You can comment out now that pip install we did for the requirements.txt.

OK, now for the whole project, since this whole projects is really big and there will be so many things to go over, I believe it's best that you clone the repository I had on github and most importantly once you cloned the repository, just go on the data-processing folder, and look at the data-proccessing.ipynb file, you can view it by downloading it into google drive, the same way you did with the requirements.txt file, and just follow along what i did, i have comments, text and code that will guide you there. I believe this is the most straight forward method to show you how to run a LLM without taking so long and that you most importantly figure out things on by your own as that's what makes you a better developer along side working with others of course.
Right now I will explain key points and an overview what you will get from this project.

In this whole project we will go over how to use RAG with Gemma-3b-ints float16 at first but than use others models like gpt-3.5 turbo, gpt-4o, llama3, xgen and bloom. Gemma can be a great model to start as it's not big but not very small that it's useless. Now let me go over what this 3b stands for. In the LLM's there are parameters, these parameters are also seen in Deep Learning as the Weights & Bias of the model and of course when i say "model" I'm referring to the base Large Language Model (LLM) or Algorithm but more exactly LLM. 

Let's clear up Parameters: Inside a LLM there is a neural network and in this there are layers and in layers there are neurons. Each neuron finds a similarity with itself onto another neuron across another layer, these similarities are done with the interconnection of each neuron but that connection has relevant amount of feedback towards how similar these neurons are to each other and the connection between them. Depending on the similarity becomes weighted, the more weight there is on the connection, the more it will influence that neuron and that neuron onto another neuron. That's for weights.. But what about Bias.
Bias is actually done at the training process like the weights but there set to random values onto each neuron to effect a different response to say looking at the same input data as each neuron in the whole network does in each swing, this is very important, because that's what gives the model really a generalizing and say that intelligence that ordinary people see when it can respond to different subjects.. To be more specific:

## Initialization:
* Biases (and weights) are initially set to small random values or zeros.
* This randomness helps to prevent symmetry, allowing different neurons to learn different features. (features are characteristics and patterns it finds in the input data, each neuron grabs a different features because it's value is set differently from the bias but also to the weights.)

1. Initial Forward Pass:
* With these initial random biases and weights, the network's initial predictions to the outputs are varied and likely inaccurate.

However this is where the model than corrects its inaccuracy, by later using the gradients during backpropagation.

## Training Process
2. Forward Pass:
* Inputs are fed through the network, and each neuron's output is calculated using its current weights and biases.
* For each neuron: 𝑧=∑𝑖𝑤𝑖𝑥𝑖+𝑏z=∑i​wi​xi​+b
* The activation function (e.g., sigmoid, ReLU) is applied to z to get the final output.

3. Loss Calculation:
* The network's outputs are compared to the actual target values using a loss function.
* The loss function quantifies how far off the network's predictions are from the true values.

4. Backpropagation:
* The loss is propagated back through the network to calculate gradients for each weight and bias.
* Gradients indicate the direction and magnitude of change needed to reduce the loss.

5. Gradient Descent:
* Weights and biases are updated using the calculated gradients
> 𝑤𝑖=𝑤𝑖−𝜂∂𝐿∂𝑤𝑖wi​=wi​−η∂wi​∂L​
> 𝑏=𝑏−𝜂∂𝐿∂𝑏b=b−η∂b∂L​
* This process adjusts the weights and biases to minimize the loss.

During training, biases (along with weights) are adjusted systematically through backpropagation and gradient descent. This process aims to minimize the loss function, which measures how far the network's predictions are from the actual targets.

## Convergence and Bias Adjustment
6. Convergence:
* As training progresses, weights and biases are adjusted iteratively.
* The network's predictions become more accurate, and the loss decreases.

7. Bias and Weight Adjustment:
* Both weights and biases are fine-tuned to reduce the error in predictions.
* The randomness from the initial biases is reduced as the biases are adjusted to optimal values from the gradients, that help improve the network's accuracy.

8. Reduced Randomness:
* The adjustment of biases (and weights) during training reduces the initial randomness.
* This helps the network stabilize and produce consistent, accurate outputs.

As training progresses, biases and weights converge to values that help the network make accurate predictions. The randomness from initialization is gradually reduced, and the network becomes more stable and focused.


I know this can be a bit complicated and overwhelming, but let me explain it in layman terms, really the bias and weights shine in 2 areas. 

1) Is during the training process were the bias is set to different random values on each neuron so that neuron can retrieve a different characteristics or feature on the input data compared to other neurons, since that neuron had a different value that was placed with the bias and was effected on the connection from the weight, that will make it generalize more on the output and create a response with more generality.

2) Is during backpropagation, when the gradient looks at the bias, it will systematically adjust the value again based on the loss, so in other words, the bias serves as viewer to a problem that the gradient as tool can adjust the bias value based on the loss than can be seen. This will make the gradient adjusted those value and reduce loss being shown by the bias on each neuron.

Now that you understand parameters, you can understand how a model that has 140 Billion parameters can give a more concise and general response compared to a 8–40 billion parameters model, because these parameters capture different features and give and outlet to fixing more errors that around seen from a few points. However, the more parameters a model has the more computation is needed since the connection and overall transfer of data between is so much more frequent and bigger in scale which makes the hardware need to work faster and also with more tasks.

I highly recommend, if you want to understand more how this process works such as how a LLM can receive input and calculate -> to making output exactly in a more visual way, view these videos below:
How GPT works: https://youtu.be/wjZofJX0v4M?si=nsAQAprPuT2QEqZK
Neural Networks: https://youtu.be/aircAruvnKk?si=pNLQM6IytgaYlUBF
