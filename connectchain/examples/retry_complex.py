"""This example demonstrates how to use the LCELRetry class to retry a chain of runnables in case of
exceptions. The example simulates a database connection that may fail, and a model that may fail
due to a firewall issue or a timeout.
"""
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda, Runnable

from connectchain.lcel import LCELRetry, model


load_dotenv(find_dotenv())

class DBException(Exception):
    """Custom exception for database failures."""

class FirewallException(Exception):
    """Custom exception for firewall failures."""

class ModelTimeoutException(Exception):
    """Custom exception for model timeout failures."""

n_db_failure = 0
def simulated_database_connection(user_id):
    global n_db_failure
    n_db_failure += 1
    if n_db_failure < 3:
        raise DBException(f'Simulated DB failure for user {user_id}')
    return { 'species': 'fish' }


class SimulatedUnreliableModel(Runnable):
    def __init__(self, model_id):
        self.failures = 0
        self.model = model(model_id)

    def invoke(self, input):
        self.failures += 1
        if self.failures == 1:
            raise FirewallException('Simulated firewall failure')
        if self.failures == 2:
            raise ModelTimeoutException('Simulated Model timeout')
        return self.model.invoke(input)

prompt = PromptTemplate(input_variables=['species'], template='Tell me about {species}.')

chain = (
    LCELRetry(
        RunnableLambda(lambda user_id: simulated_database_connection(user_id)),
        max_retry=3,
        sleep_time=1,
        exceptions=DBException,
        ebo=True
    )
    | prompt
    | LCELRetry(
        SimulatedUnreliableModel('2'),
        max_retry=3,
        sleep_time=1,
        exceptions=[FirewallException, ModelTimeoutException]
    )
    | StrOutputParser()
)
res = chain.invoke('user_1234')
print(res)
