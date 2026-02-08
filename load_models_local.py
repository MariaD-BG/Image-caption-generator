from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

model_name = "openai/clip-vit-base-patch32"
save_directory = "./local_clip_model"

print(f"Downloading {model_name}...")

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
tokenizer = CLIPTokenizer.from_pretrained(model_name)

model.save_pretrained(save_directory)
processor.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model saved to {save_directory}")