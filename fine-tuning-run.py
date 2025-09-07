import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load your fine-tuned model checkpoint
model_path = "./deberta-finetuned-classifier"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

statements = [
    "Wolfgang Amadeus Mozart composed over 600 works, many of which are considered classics of Western classical music.",
    "The global banking system is expected to fully transition to a decentralized model by 2025, fundamentally changing how financial transactions are conducted and managed, according to recent industry predictions.",
    "The city of Venice is built on a series of islands and has more than 400 bridges.",
    "The city of Venice is built on a series of islands and has more than 400 bridges.",
    "The city of Venice is built on a series of islands and has more than 400 bridges.",
    "The city of Venice is built on a series of islands and has more than 400 bridges.",
    "The city of Venice is built on a series of islands and has more than 400 bridges.",
    "The city of Venice is built on a series of islands and has more than 400 bridges.",
    "The city of Venice is built on a series of islands and has more than 400 bridges.",
    "The city of Venice is built on a series of islands and has more than 400 bridges.",
    "The city of Venice is built on a series of islands and has more than 400 bridges.",
    "The city of Venice is built on a series of islands and has more than 400 bridges.",
    "a growing young adult male base classifying the music, especially boy band music, as effeminate, and · other musical genres began increasing in popularity. 1990s and early 2000s teen pop artists eventually entered hiatuses and semi-retirements (*NSYNC, Dream, Destiny's Child) or changed their musical style, including the Backstreet Boys, Britney Spears, Christina Aguilera, Jessica Simpson, Mandy Moore, 3LW and Aaron Carter.",
]
excerpts = [
    "Despite his short life, his rapid pace of composition and proficiency from an early age resulted in more than 800 works representing virtually every Western classical genre of his time. Many of these compositions are acknowledged as pinnacles of the symphonic, concertante, chamber, operatic, and choral repertoires. Mozart is widely regarded as one of the greatest composers in the history of Western music, with his music admired for its \"melodic beauty, its formal elegance and its richness of harmony and texture\".",
    "While Decentralized Finance (DeFi) holds immense potential to revolutionize the financial landscape, several significant challenges hinder its mainstream adoption. These challenges span technological, regulatory, and operational domains, posing risks that must be addressed for DeFi to achieve widespread acceptance.",
    "Venice is the capital of the Veneto Region and is located in the northeastern part of the country. It comprises 118 small islands separated by canals and linked by over 400 bridges.",
    "Venice is the capital of the Veneto Region and is located in the northeastern part of the country. It comprises 118 small islands separated by canals and linked by over 400 bridges.",
    "Venice is the capital of the Veneto Region and is located in the northeastern part of the country. It comprises 118 small islands separated by canals and linked by over 400 bridges.",
    "Venice is the capital of the Veneto Region and is located in the northeastern part of the country. It comprises 118 small islands separated by canals and linked by over 400 bridges.",
    "Venice is the capital of the Veneto Region and is located in the northeastern part of the country. It comprises 118 small islands separated by canals and linked by over 400 bridges.",
    "Venice is the capital of the Veneto Region and is located in the northeastern part of the country. It comprises 118 small islands separated by canals and linked by over 400 bridges.",
    "Venice is the capital of the Veneto Region and is located in the northeastern part of the country. It comprises 118 small islands separated by canals and linked by over 400 bridges.",
    "Venice is the capital of the Veneto Region and is located in the northeastern part of the country. It comprises 118 small islands separated by canals and linked by over 400 bridges.",
    "Venice is the capital of the Veneto Region and is located in the northeastern part of the country. It comprises 118 small islands separated by canals and linked by over 400 bridges.",
    "Venice is the capital of the Veneto Region and is located in the northeastern part of the country. It comprises 118 small islands separated by canals and linked by over 400 bridges.",
    "Pop music is primarily dominated by male artists, with female representation decreasing significantly in the last decade.",
]

start_time = time.time()

# Batch tokenize
inputs = tokenizer(statements, excerpts, padding=True, truncation=True, return_tensors="pt")

# Run forward pass
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

print(predictions)  # e.g. tensor([0, 1]) but now from your fine-tuned model

print(f"Elapsed time: {time.time() - start_time}")
