"""This example demonstrates how to use the LCELRetry class to retry a failed chain."""
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda

from connectchain.lcel import LCELRetry, model


load_dotenv(find_dotenv())

EXAMPLE_INPUT = { 'species': 'birds' }

n_failure = 0
def simulated_failure(input):
    global n_failure
    n_failure += 1
    if n_failure < 3:
        raise Exception('Simulated failure')
    return model('2').invoke(input)

prompt = PromptTemplate(
    input_variables=['species'], template='Tell me about the a biggest {adjective} animal in the world.'
)

try:
    failed_chain = (
        prompt
        | RunnableLambda(lambda x: simulated_failure(x))
        | StrOutputParser()
    )
    res = failed_chain.invoke(EXAMPLE_INPUT)
except Exception as e:
    print(f'Failed to invoke the model: {e}')

chain = (
    prompt
    | LCELRetry(RunnableLambda(lambda x: simulated_failure(x)))
    | StrOutputParser()
)
res = chain.invoke(EXAMPLE_INPUT)
print(res)
