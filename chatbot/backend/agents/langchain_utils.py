from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

def get_clean_chain():
    prompt = PromptTemplate.from_template(
        """Respond to this query in STRICT MARKDOWN FORMAT (NO JSON, NO CODE BLOCKS):
        
        Query: {query}
        Mode: {mode}
        
        Follow this structure EXACTLY:
        # [Query Topic]  
        ## Summary  
        [1-paragraph summary]  
        ## Key Points  
        - [Point 1]  
        - [Point 2]  
        - [Point 3]  
        ## Sources  
        1. [Source 1]  
        2. [Source 2]  

        NEVER use: {{}}, ```, or any other formatting."""
    )
    
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.5  # Lower for more structured output
    )
    
    return prompt | llm | StrOutputParser()