# These chains are focused on improving the summary/overview section

from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import SequentialChain


def improve_summary_chain() -> SequentialChain:
    """ Customizable chain to improve then summarize any resume section.

    Notes:
    Future iterations might benefit from being given a list of required skills instead of a job description.

    Inputs:
    section: resume section to process
    desc: job description to adapt for
    format (optional): summary grammatical structure (eg: bullets, one paragraph, 3 bullets and 3 sentences, etc.)

    Outputs:
    summarized: new resume section
    """
    llm = ChatOpenAI(temperature=.2, model_name="gpt-3.5-turbo")

    appeal_prompt = PromptTemplate.from_template("""
    You will be given a resume section. You will also be given a job description.
    Improve the given resume section to be more appealing to a recruiter looking to fill the given job description.
    The goal is to convey confidence and competence.
    Return only the improved summary, with no header.
    
    Resume Section:
    {section}
    
    Job Description:
    {desc}
    """)
    appeal_chain = LLMChain(prompt=appeal_prompt, llm=llm, output_key="improved")

    summ_prompt = PromptTemplate.from_template("""
    Condense the following resume summary section using {format}.
    The summary should remain appealing to a recruiter and be less than 260 characters.
    
    Resume section:
    {improved}
    """,
                                               partial_variables={'format': 'a mix of bullets and 1 short paragraph'})
    summ_chain = LLMChain(prompt=summ_prompt, llm=llm, output_key="summarized")

    return SequentialChain(
        chains=[appeal_chain, summ_chain],
        input_variables=["section", "desc"],
        output_variables=["summarized"],
    )
