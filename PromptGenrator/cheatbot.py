import os
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import HumanMessage

os.environ["HF_HOME"] = "D:/huggingface_cache"

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "temperature": 0.3,
        "do_sample": True,
    },
)

model = ChatHuggingFace(llm=llm)

while True:
    user_input = input("You :- ")

    if user_input.lower() == "exit":
        break
      
    result = model.invoke([HumanMessage(content=user_input)])
    print("AI :-", result.content)
