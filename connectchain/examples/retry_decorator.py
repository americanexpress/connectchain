"""Example of using retry decorator."""
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

from connectchain.lcel import model
from connectchain.utils import retry_decorator


load_dotenv(find_dotenv())

n_failure = 0

@retry_decorator()
def simulated_failure(input):
    global n_failure
    n_failure += 1
    if n_failure < 2:
        raise Exception('Simulated failure')
    prompt = PromptTemplate(
        input_variables=['species'],
        template='What is your favorite {species}?'
    )
    return (prompt | model('2') | StrOutputParser()).invoke(input)

res = None

while not res:
    res = simulated_failure({ 'species': 'mammal' })

print(res)
