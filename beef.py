""" Chains for improving a job experience section when it is not strong for a given job description / title """
from langchain import LLMChain, PromptTemplate
from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser


def get_key_skill() -> LLMChain:
    """ Extract skills relevant to job description from resume section """
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
                    output_parser=output_parser)


def _summary_sentence_from_skills() -> LLMChain:
    """ Generate a summary sentence showing how a job experience is demonstrates the given skills.

    This produces a nicely worded, concise summary. This might be useful in the future,
    but is not planned in be implemented now.
    """
    prompt = PromptTemplate.from_template("""
    You will be given a section of a resume and 3 key skills.
    Please write a one sentence summary for the resume section highlighting some or all of the given skills.
    
    Skills: {skills}
    
    Section: {section} 
    """)
    return LLMChain(prompt=prompt, llm=ChatOpenAI(temperature=.1, model_name="gpt-3.5-turbo"))


def highlight_chain() -> LLMChain:
    """ Frame weak job experience section to highlight key skills """

    prompt = PromptTemplate.from_template("""
    You will be given a job experience from a resume and 3 key skills.
    Use bullet points to highlight how the job experience section demonstrates the given skills.
    
    Skills: {skills}
    
    Job Experience: {section} 
    """)
    return LLMChain(prompt=prompt, llm=ChatOpenAI(temperature=.1, model_name="gpt-3.5-turbo"))


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

    # chain to format section as markdown
    format_prompt = PromptTemplate.from_template("""
    May you please reformat the following experience from a resume using the following format:

    ```
    ## Job Title, Company, Dates (Total time)

        experience description
        
        - bullet points
    ```

    Here is experience text:
    \n{emulated}
    """)
    format_chain = LLMChain(prompt=format_prompt,
                            llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
                            output_key="formatted")

    return SequentialChain(
        chains=[grammatical_chain, format_chain],
        input_variables=["title", "desc", "section"],
        output_variables=["formatted"],
        verbose=True,)
