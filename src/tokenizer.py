from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainerState
from const import Constant 

class Tokenizer:
    def __init__(self):
        self.model = None
        self.tokenizer = None


    def create_token(self, model_name:str):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)


    def add_prefix(self, articles):
        inputs = ["summarize": + document for document in articles["article"]]
        model_token_input = self.tokenizer(
            inputs, 
            max_length=Constant().max_input_len, 
            truncation=True,
            padding="max_length",
            return_tensors="pt")
         
        labels = self.tokenizer(
            text_target=articles["highlights"], 
            max_length=Constant().max_output_len, 
            truncation=True,
            padding="max_length",
            return_tensors="pt")
        
        model_token_input["labels"] = labels["input_ids"]
        return model_token_input