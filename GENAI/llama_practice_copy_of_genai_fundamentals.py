"""
# Generative AI & Large Language Models (LLMs)

## Objectives
In this notebook, you will learn:
- What is Generative AI and LLMs
- Transformer architecture & self-attention
- BERT for text classification (Sentiment Analysis)
- GPT for text generation
- T5 for sequence-to-sequence tasks (Summarization)
- Hugging Face ecosystem
- Open-source LLMs
- Prompt Engineering fundamentals & practice
- Understanding model limitations (Hallucinations)

## What is Generative AI?

Generative AI refers to models that can **generate new content**:
- Text
- Images
- Code
- Audio

Unlike traditional ML (predict/classify), GenAI **creates**.

### Examples:
- ChatGPT → text generation
- DALL·E → image generation
- GitHub Copilot → code completion

## Large Language Models (LLMs)

LLMs are deep learning models trained on massive text corpora.

Examples:
- BERT → Understanding-focused
- GPT → Generation-focused
- T5 → Text-to-text framework

Key properties:
- Billions of parameters
- Trained using self-supervised learning
- Use Transformer architecture

## Transformer Architecture

Transformers replaced RNNs/LSTMs by enabling:
- Parallel processing
- Long-range dependency capture

### Core Components:
- Token Embeddings
- Positional Encoding
- Multi-Head Self-Attention
- Feed Forward Networks

## Attention Mechanism (Self-Attention)

Self-attention allows each word to **attend to all other words**.

It computes:
- Query (Q)
- Key (K)
- Value (V)

This helps the model understand **context** and **relationships**.
"""

#!pip install transformers datasets sentencepiece accelerate

"""## Hugging Face Ecosystem

Key components:
- `transformers` → Models & pipelines
- `tokenizers` → Fast tokenization
- `datasets` → Ready-to-use datasets
- `hub` → Model sharing & versioning
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

"""## BERT (Bidirectional Encoder Representations from Transformers)

BERT is designed for **language understanding**:
- Reads text bidirectionally
- Excels at classification, NER, QA

We will use BERT for **Sentiment Analysis**.
"""

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

#sentiment_pipeline("The product quality is amazing and delivery was fast.")
sentiment_pipeline("Worst product ever")

"""## GPT (Generative Pre-trained Transformer)

GPT models are:
- Autoregressive
- Excellent at text generation & completion

Used for:
- Chatbots
- Story writing
- Code generation
"""

generator = pipeline("text-generation", model="gpt2")

out=generator(
    "Artificial Intelligence will transform the future of",
    max_new_tokens=50,
    pad_token_id=50256
)
print(out[0])

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

output = generator(
    "Artificial Intelligence is",
    max_new_tokens=50,
    pad_token_id=50256
)

print(output[0]['generated_text'])

"""## T5 (Text-to-Text Transfer Transformer)

T5 treats **every NLP task as text → text**:
- Translation
- Summarization
- QA
"""

#!pip install -U transformers accelerate sentencepiece

summarizer = pipeline(
    "text-generation",
    model="t5-small"
)

text = """
Generative AI models have revolutionized the field of artificial intelligence by enabling machines to generate human-like text.
These models are widely used in chatbots, content generation, and search engines.
"""

summarizer(text, max_length=50)

from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

text = "Summarize: Generative AI is transforming industries..."

output = generator(text, max_new_tokens=50)

print(output[0]['generated_text'])

from transformers.pipelines import SUPPORTED_TASKS
print(SUPPORTED_TASKS.keys())

"""## Pretrained Models Summary

| Model | Best For |
|------|---------|
| BERT | Understanding, classification |
| GPT | Text generation |
| T5 | Seq-to-seq tasks |

Using pretrained models saves:
- Time
- Compute
- Data requirements

## Open-Source LLMs

Examples:
- LLaMA (Meta)
- Mistral
- Falcon
- BLOOM

Benefits:
- Transparency
- Custom fine-tuning
- On-prem deployment

## Prompt Engineering

Prompt engineering is the art of **designing effective inputs** for LLMs.

Good prompts:
- Are clear
- Provide context
- Specify format

## Prompting Techniques

1. **Zero-shot Prompting**
   - No examples

2. **One-shot Prompting**
   - One example

3. **Few-shot Prompting**
   - Multiple examples

## Model Limitations – Hallucinations

LLMs may:
- Generate incorrect facts
- Sound confident but be wrong

Mitigation:
- Grounding with data
- Retrieval-Augmented Generation (RAG)
- Explicit constraints in prompts

## Popular LLM Providers

- **OpenAI** → GPT-5.2, GPT-5.1
- **Anthropic** → Claude Opus
- **Google** → Gemini 3

Each differs in:
- Safety
- Cost
- Context length

## Prompt Engineering Practice (Groq)

Use any Groq Model (API) to try:

### Summarizing
"Summarize the following text in 5 bullet points."

Paragraph - When Percy Jackson gets an urgent distress call from his friend Grover, he immediately prepares for battle. He knows he will need his powerful demigod allies at his side, his trusty bronze sword Riptide, and… a ride from his mom.

The demigods rush to the rescue to find that Grover has made an important discovery: two powerful half-bloods whose parentage is unknown. But that’s not all that awaits them. The titan lord Kronos has devised his most treacherous plot yet, and the young heroes have just fallen prey.

The're not the only ones in danger. An ancient monster has arisen — one rumored to be so powerful it could destroy Olympus — and Artemis, the only goddess who might know how to track it, is missing. Now Percy and his friends, along with the Hunters of Artemis, have only a week to find the kidnapped goddess and solve the mystery of the monster she was hunting. Along the way, they must face their most dangerous challenge yet: the chilling prophecy of the titan’s curse.

### Inferring
"What can be inferred about reader's sentiment?"

### Transforming
"Convert this paragraph into a movie script."

### Expanding
"Expand this idea into a detailed explanation."

### Few-shot Example
Provide 2 examples before asking the task.

### Summarize the following text in 5 bullet points.
"""

from groq import Groq

client = Groq(api_key="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")

paragraph="""

When Percy Jackson gets an urgent distress call from his friend Grover, he immediately prepares for battle. He knows he will need his powerful demigod allies at his side, his trusty bronze sword Riptide, and… a ride from his mom.

The demigods rush to the rescue to find that Grover has made an important discovery: two powerful half-bloods whose parentage is unknown. But that’s not all that awaits them. The titan lord Kronos has devised his most treacherous plot yet, and the young heroes have just fallen prey.

They're not the only ones in danger. An ancient monster has arisen — one rumored to be so powerful it could destroy Olympus — and Artemis, the only goddess who might know how to track it, is missing. Now Percy and his friends, along with the Hunters of Artemis, have only a week to find the kidnapped goddess and solve the mystery of the monster she was hunting. Along the way, they must face their most dangerous challenge yet: the chilling prophecy of the titan’s curse."
"""
prompt = f"Summarize the given paragraph in 5 bullet points: {paragraph}"

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)

"""### Inferring
  - What can be inferred about reader's sentiment?
"""

from groq import Groq

client = Groq(api_key="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")

pargraph="""
 When Percy Jackson gets an urgent distress call from his friend Grover, he immediately prepares for battle. He knows he will need his powerful demigod allies at his side, his trusty bronze sword Riptide, and… a ride from his mom.

The demigods rush to the rescue to find that Grover has made an important discovery: two powerful half-bloods whose parentage is unknown. But that’s not all that awaits them. The titan lord Kronos has devised his most treacherous plot yet, and the young heroes have just fallen prey.

They're not the only ones in danger. An ancient monster has arisen — one rumored to be so powerful it could destroy Olympus — and Artemis, the only goddess who might know how to track it, is missing. Now Percy and his friends, along with the Hunters of Artemis, have only a week to find the kidnapped goddess and solve the mystery of the monster she was hunting. Along the way, they must face their most dangerous challenge yet: the chilling prophecy of the titan’s curse."
"""
prompt = f"""Read the following paragraph carefully.

What can be inferred about:
- The situation
- The emotions involved
- The level of danger

Paragraph: {paragraph}"""

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)

"""### Transforming
  - Convert this paragraph into a movie script.
"""

from groq import Groq

client = Groq(api_key="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")

paragraph="""
 When Percy Jackson gets an urgent distress call from his friend Grover, he immediately prepares for battle. He knows he will need his powerful demigod allies at his side, his trusty bronze sword Riptide, and… a ride from his mom.

The demigods rush to the rescue to find that Grover has made an important discovery: two powerful half-bloods whose parentage is unknown. But that’s not all that awaits them. The titan lord Kronos has devised his most treacherous plot yet, and the young heroes have just fallen prey.

They’re not the only ones in danger. An ancient monster has arisen — one rumored to be so powerful it could destroy Olympus — and Artemis, the only goddess who might know how to track it, is missing. Now Percy and his friends, along with the Hunters of Artemis, have only a week to find the kidnapped goddess and solve the mystery of the monster she was hunting. Along the way, they must face their most dangerous challenge yet: the chilling prophecy of the titan’s curse."
"""
prompt = f"""Read the following paragraph carefully.

convert this paragraph into movie script

Paragraph: {paragraph}"""

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)

"""### Expanding
  - Expand this idea into a detailed explanation.
"""

from groq import Groq

client = Groq(api_key="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")

paragraph="""
 When Percy Jackson gets an urgent distress call from his friend Grover, he immediately prepares for battle. He knows he will need his powerful demigod allies at his side, his trusty bronze sword Riptide, and… a ride from his mom.

The demigods rush to the rescue to find that Grover has made an important discovery: two powerful half-bloods whose parentage is unknown. But that’s not all that awaits them. The titan lord Kronos has devised his most treacherous plot yet, and the young heroes have just fallen prey.

They’re not the only ones in danger. An ancient monster has arisen — one rumored to be so powerful it could destroy Olympus — and Artemis, the only goddess who might know how to track it, is missing. Now Percy and his friends, along with the Hunters of Artemis, have only a week to find the kidnapped goddess and solve the mystery of the monster she was hunting. Along the way, they must face their most dangerous challenge yet: the chilling prophecy of the titan’s curse."
"""
prompt = f"""Read the following paragraph carefully.

Expand this idea into a detailed explanation

Paragraph: {paragraph}"""

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)

"""###  Few-shot Example
  - Provide 2 examples before asking the task.
"""

from groq import Groq

client = Groq(api_key="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")

prompt = """
Input: A boy saved a dog from drowning in a river and became a hero.
Output:
- A boy rescued a dog
- The dog was drowning
- The boy became a hero

Example:
Input: A massive storm hit the coastal town causing floods and damage.
Output:
- A storm struck a coastal town
- Flooding occurred
- Property was damaged

Now summarize:
Input: When Percy Jackson gets an urgent distress call from his friend Grover, he immediately prepares for battle. He knows he will need his powerful demigod allies at his side, his trusty bronze sword Riptide, and a ride from his mom. The demigods rush to rescue Grover, who has discovered two powerful half-bloods of unknown origin. Meanwhile, Kronos has set a dangerous trap, and an ancient monster capable of destroying Olympus has risen. Artemis is missing, and Percy must find her within a week while facing a dangerous prophecy.
Output:
"""

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)

"""## Practice Exercises

1. Perform sentiment analysis on 5 customer reviews using BERT/Any Groq Model.
2. Generate a product description using Llama.
3. Summarize a news article using GPT-OSS.
4. Try zero-shot vs few-shot prompting and compare outputs.
5. Identify hallucinations in generated content.

### Perform sentiment analysis on 5 customer reviews using BERT
"""

from transformers import pipeline

classifier = pipeline("sentiment-analysis")

reviews = [
    "The product quality is amazing and delivery was fast.",
    "Very bad experience, the item was damaged.",
    "It’s okay, not great but not terrible.",
    "Excellent service! Highly recommend.",
    "Waste of money, very disappointed."
]

results = classifier(reviews)

for review, result in zip(reviews, results):
    print(f"Review: {review}")
    print(f"Sentiment: {result['label']}, Confidence: {result['score']:.2f}")
    print("-"*50)

#!pip install groq

"""### Generate a product description using Llama."""

from groq import Groq

client = Groq(api_key="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")

product = "Smart Fitness Watch"

prompt = f"""
Act as a professional product marketer.

Write a compelling product description for:
Product: {product}

Include:
- Key features
- Benefits
- Target audience
Tone: professional and engaging
Length: 100-120 words
"""

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)

"""### Identify hallucinations in generated content.

  - Heart rate monitoring and ECG tracking
  - Built-in GPS

### Summarize a news article using GPT-OSS.
"""

from groq import Groq

"""
major technology company announced a breakthrough in artificial intelligence, introducing a new model that significantly improves efficiency and reduces costs. Experts believe this
innovation could transform industries such as healthcare, finance, and education. However, concerns remain about ethical implications and job displacement.
"""

client = Groq(api_key="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")

article = input("Paste your article here:\n")

prompt = f"""
Summarize the following news article in 4 bullet points:
{article}

Include:
- Features
- Benefits
- Use cases
- Attractive tone
"""

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)

from groq import Groq

client = Groq(api_key="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")

def chat(user_input):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": user_input}
        ]
    )

    return response.choices[0].message.content


while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    reply = chat(user_input)
    print("Bot:", reply)

"""### Try zero-shot vs few-shot prompting and compare outputs."""

from groq import Groq


client = Groq(api_key="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")

text = "The product is okay, not great but not bad."

# Zero-shot Prompt
zero_shot_prompt = f"""
Classify sentiment (Positive, Negative, Neutral):

"{text}"
"""

# Few-shot Prompt
few_shot_prompt = f"""
Example:
Input: I love this product
Output: Positive

Input: This is terrible
Output: Negative

Input: It's okay, nothing special
Output: Neutral

Now classify:
Input: {text}
Output:
"""

def get_response(prompt, model):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

models = [
    "llama-3.1-8b-instant"
]

for model in models:
    print(f"\n🔹 Model: {model}")

    zero_output = get_response(zero_shot_prompt, model)
    few_output = get_response(few_shot_prompt, model)

    print("Zero-shot Output :", zero_output)
    print("Few-shot Output  :", few_output)
    print("-" * 50)

