from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """
    You are an expert AI assistant. Answer the following questions based strictly on the provided context. Do not use any external knowledge or assumptions.

    Context:
    {context}

    Questions:
    {input}

    Please ensure that your answers are:
    - Accurate
    - Complete
    - Directly related to the context provided

    Provide the most accurate and contextually relevant response to each question.
    """
)
