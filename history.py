""" Chains to improve job experience sections """

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


def generate_star_chain() -> LLMChain:
    """ Convert statements inside a job experience section to STAR statements.

    STAR = Situation, Task, Action, Result. LLMs seem to hallucinate to when a Result clause is not included in original
    content.

    'star' is the output key for this chain.
    """

    # TODO: it might be profitable to fine-tune Result statements to emphasize key skills instead letting
    #  hallucination take ever

    prompt = PromptTemplate(template="""
    You will be given a job experience section from a resume. Your task is to convert extract as many achievement
    statements as possible using the STAR method.


    The STAR method helps discuss how skills were used to achieve goals.
    STAR stands for Situation, Task, Action, and Result.
    It uses brief examples that give a fuller picture of competencies.
    An achievement statement describes how well a task was performed. In developing STAR statements, emphasize where job
    requirements were exceeded to help stand out.
    In a STAR statement, most of the content is from the Action and Result sections. The statement began with a powerful
    action verb and used numbers to quantify the accomplishment, but should not exaggerate.


    This is an example of a STAR statement:
    "Developed and applied a comprehensive document tracking system, ensuring that 100% of 5,500 promotion packages were
    updated, correct, and completed ahead of the Promotion Board deadline."
    DO NOT use this example in the output for any reason.


    Here is the job experience section:
    `{section}`


    The response should be a statement, and not S, T, A, R.

    """,
                            input_variables=['section'])
    return LLMChain(prompt=prompt, llm=ChatOpenAI(temperature=0.2,
                                                  model_name='gpt-3.5-turbo'), output_key='star')
