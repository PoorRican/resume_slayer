""" Chains for improving a job experience section when it is not strong for a given job description / title """
from langchain import LLMChain, PromptTemplate
from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import SimpleMemory
from langchain.output_parsers import CommaSeparatedListOutputParser

from util import job_requirement_chain


def get_key_skill() -> LLMChain:
    """ Extract skills relevant to job description from resume section.

    Inputs:
    section: job history section from resume
    requirements: list of requirements extracted from job description. eg: `util.job_requirements_chain()`

    Outputs:
    list of relevant skills mentioned in resume section
    """
    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(template="""
    You're an expert career consultant with an IQ over 140 working with a special client regarding this job posting.
    You will be given a list of required skills, and an excerpt from the client's resume.
    What required 3 skills does the client have?
    
    Resume excerpt:\n\t{section}
    
    Required skills:\n\t{requirements}
    
    {format_instructions}
    """,
                            input_variables=['section', 'requirements'],
                            partial_variables={'format_instructions': format_instructions}
                            )
    return LLMChain(prompt=prompt,
                    llm=ChatOpenAI(temperature=.1,
                                   model_name="gpt-3.5-turbo"),
                    output_parser=output_parser,
                    output_key='skills')


def _summary_sentence_from_skills() -> LLMChain:
    """ Generate a summary sentence showing how a job experience is demonstrates the given skills.

    This produces a nicely worded, concise summary. This might be useful in the future,
    but is not planned in be implemented now.

    Inputs:
    skills: list of 3 requirements extracted from job description. eg: `util.job_requirements_chain()`
    section: job history section from resume

    Outputs:
    a 1-sentence summary than nicely describes the job history using given points. This summary omits anything
    that does not pertain to skills.
    """
    prompt = PromptTemplate.from_template("""
    You will be given a section of a resume and 3 key skills.
    Please write a one sentence summary for the resume section highlighting some or all of the given skills.
    
    Skills: {skills}
    
    Section: {section} 
    """)
    return LLMChain(prompt=prompt, llm=ChatOpenAI(temperature=.1, model_name="gpt-3.5-turbo"),
                    output_key='summary')


def highlight_chain() -> LLMChain:
    """ Frame weak job experience section to highlight key skills.

    Notes:
    - Structures the entire job history section under the given 3 skills.
    - Might be wordy, but tends to include more details than `_summary_sentence_from_skills()`

    Inputs:
    skills: list of 3 requirements extracted from job description. eg: `util.job_requirements_chain()`
    section: job history section from resume

    Outputs:
    Job history section is restructured under the 3 key skills as bullet points. Most of the original data is retained.
    """

    prompt = PromptTemplate.from_template("""
    You will be given a job experience from a resume and 3 key skills.
    Use bullet points to highlight how the job experience section demonstrates the given skills.
    
    Skills: {skills}
    
    Job Experience: {section} 
    """)
    return LLMChain(prompt=prompt,
                    llm=ChatOpenAI(temperature=.1, model_name="gpt-3.5-turbo"),
                    output_key="highlighted")


def format_chain() -> LLMChain:
    # chain to format section as markdown
    format_prompt = PromptTemplate.from_template("""
    May you please reformat the following experience from a resume using the following format:

    ```
    ## Job Title, Company, Dates (Total time)

        - bullet points
    ```

    Here is experience text:
    \n{highlighted}
    """)
    return LLMChain(prompt=format_prompt,
                    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
                    output_key="formatted")


def process_history_chain() -> SequentialChain:
    # chain emulate section wording and grammar
    gram_prompt = PromptTemplate.from_template("""
        You're an expert career consultant with an IQ over 140 working with a special client regarding this job posting.
        Please improve this resume section for this {title} position.\n
        Improve the section by matching grammatical syntax and lexicon.

        This is the job description:\n\n{desc}.
        \n\n
        Here is the resume section:\n{section}
    """)
    grammatical_chain = LLMChain(prompt=gram_prompt,
                                 llm=ChatOpenAI(temperature=.85, model_name="gpt-3.5-turbo"),
                                 output_key="emulated"
                                 )

    return SequentialChain(
        chains=[grammatical_chain, format_chain()],
        input_variables=["title", "desc", "section"],
        output_variables=["formatted"],
        verbose=True,)
