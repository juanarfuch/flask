from langchain.prompts import PromptTemplate    

CONDENSE_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:""") 

template =""" You are a helpful AI assistant. Use the following pieces of context delimited by << >> to answer the question at the end.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context. The context is from a vid
<<{context}>>
Question: {question}
Helpful answer in markdown:')
"""

QA_PROMPT=PromptTemplate(template=template, input_variables=["question", "context"])