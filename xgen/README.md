### Xgen from Salesforce

I wanted to spice a few things up using an other transformer from hugging face, apparently this one it factual in it's response but also creative..

You should know that whenever you use an open source model from hugging face sometimes the only way to get access to the model is by getting a license agreement from the developers.

Don't worry though this is very easiy actually, for hugging face, since all these models are in hugging face besides the GPT's. When you go to this link: https://huggingface.co/Salesforce/xgen-7b-8k-inst

Which is the exact model for this Xgen, you will see immediately if it's your first time there, they will show you a license agreement you can submit, once you submit that your just say an independent researcher or your company, that will let you get access to it. They just do this so there won't be any bots around trying to get hold of the model immensely which would cost a lot of $.

I just used a pretty practicaly configuration and instance in the app.py to get the model working for generating and displaying on gradio, if you want to add more things in there such as vision or audio, you can!