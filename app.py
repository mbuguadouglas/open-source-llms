from langchain_ollama import OllamaLLM


# model = OllamaLLM(model='gemma:2b')
model = OllamaLLM(model='qwen2:0.5b')

result = model.invoke(input='Why is the sky blue?')

print(result)