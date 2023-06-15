"""This agent allows the user to extract end balance from a statement provided as a URL.
"""

import fixieai
import requests

BASE_PROMPT = """I am a simple agent that allows the user to ask questions about a web page.
I ask Func[load_doc] to load the resource from a URL, and then always reach out to
Func[fixie_query_embed] to answer questions about the resource."""

FEW_SHOTS = """
Q: What is the end balance in: http://some_url
Ask Func[load_doc]: http://some_url
Func[load_doc] says: #doc1
Ask Func[fixie_query_embed]: What is the end balance in this document #doc1
Func[fixie_query_embed] says: The end balance found in the resource
A: The end balance found in the resource

Q: I want to ask questions about this statement: http://some_url
Ask Func[load_doc]: http://some_url
Func[load_doc] says: #doc1
A: I've finished reading the resource. Go ahead and ask me about it.
Q: What is the end balance in it?
Ask Func[fixie_query_embed]: What is the end balance in #doc1
Func[fixie_query_embed] says: answer from the resource
A: answer from the resource
"""
agent = fixieai.CodeShotAgent(BASE_PROMPT, FEW_SHOTS, conversational=True)

@agent.register_func
def load_doc(query: fixieai.agents.Message) -> fixieai.Message:
    url = query.text
    response = requests.get(url)
    content_type = response.headers["Content-Type"]
    result =  fixieai.Message("#doc1", embeds={"doc1": fixieai.Embed(content_type, url)})
    # print(result.embeds["doc1"].content)
    return result