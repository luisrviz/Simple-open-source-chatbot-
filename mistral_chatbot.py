import panel as pn
from transformers import AutoTokenizer, TextStreamer
import transformers
import torch

pn.extension()

async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    #print(instance)
    #print(instance._button_data)
    messages = [{"role": "user", "content": contents}]
    chat.append({"role": "user", "content": contents})
    prompt = pipeline.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, skip_special_tokens=True)
    outputs = pipeline(prompt, max_new_tokens=500, do_sample=True, top_k=50, temperature=0.1, repetition_penalty=1.1, return_full_text=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1)
    message = ""
    for token in outputs[0]["generated_text"]:
        message += token
        yield message
    chat.append({"role": "assistant", "content": message})    
#model = "mistralai/Mistral-7B-v0.1"
model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_finetuned = "/home/lruiz/gemma/mistral-8x7b-Instruct-Forocoches-v0.1"
chat=[]
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float32, "load_in_4bit": True},
)
chat_interface = pn.chat.ChatInterface(callback=callback, callback_user="Mixtral")
chat_interface.send(
    "Send a message to get a reply from Mistral!", user="System", respond=False
)
chat_interface.servable()