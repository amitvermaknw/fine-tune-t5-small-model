from transformers import T5ForConditionalGeneration, T5Tokenizer
from const import Constant 
from load_dataset import LoadDataset

class Tokenizer:
    def __init__(self):
        self.model = None
        self.tokenizer = None


    def create_token(self, model_name:str):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)


    def add_prefix(self, articles):
        inputs = [f"summarize: {document}" for document in articles["article"]]
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
    
    @staticmethod
    def processing_dataset():
        print("Preprocessing the dataset")
        dataset = LoadDataset().load_data()
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]
        tokenized_train_dataset = train_dataset.map(Tokenizer().add_prefix, batched=True, remove_columns=["articles", "highlights", "id"])
        tokenized_validation_dataset =validation_dataset.map(Tokenizer().add_prefix, batched=True, remove_columns=["articles", "highlights", "id"])
        print("Dataset processing completed")