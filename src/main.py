import os
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

from model_training import ModelTraining

def main():
    model_training = ModelTraining()
    print("Start training.........")
    model, tokenizer  = model_training.init_training()
    print("Training completed...........")

    #Save finetune model
    os.makedirs("./models/fine_tuned", exist_ok=True)
    model.save_pretrained("./models/fine_tuned")
    tokenizer.save_pretrained("./models/fine_tuned")

if __name__ == "__main__":
    main()