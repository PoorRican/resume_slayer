from enum import Enum
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.output_parsers import PydanticOutputParser, EnumOutputParser, CommaSeparatedListOutputParser
from langchain.chains import LLMChain, SequentialChain
from langchain.text_splitter import MarkdownHeaderTextSplitter
from pydantic import BaseModel
from typing import List, Union, Optional, Generator


class Skills(BaseModel):
    skills: List[str] = []


def job_requirement_chain() -> LLMChain:
    """ Accepts job description. Returns list of skills """
    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=Skills)
    format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate(template="""
        You're an expert career consultant with an IQ over 140 working with a special client regarding this job posting.
        Please list the skills and requirements for this job.\n{format_instructions}
        Here is the job description:\n{description}
    """,
                            input_variables=["description"],
                            partial_variables={'format_instructions': format_instructions})
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    return LLMChain(prompt=prompt, llm=llm, output_parser=parser)


def chunk_markdown(resume: str) -> List[Document]:
    """ Split a resume formatted in markdown.

    Sections are split by header

    :param resume: markdown formatted resume

    :return: a list of resume sections as `Document`
    """
    headers = [
        ('#', 'Header 1'),
        ('##', 'Header 2'),
        ('###', 'Header 3'),
        ('####', 'Header 4'),
    ]
    md_spitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    return md_spitter.split_text(resume)


def cut_sections(resume: str) -> dict:
    """ Get overview / summary of resume """
    prompt = PromptTemplate.from_template("""
    May you return only the {section_name} section of this resume?
    
    {resume}
    """)

    resume_model = {'summary': {'section_name': 'resume summary',
                                'result': ''},
                    'history': {'section_name': 'job history / experience',
                                'result': ''}}
    classify_chain = LLMChain(prompt=prompt, llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
    for key in resume_model.keys():
        resume_model[key]['result'] = classify_chain.predict(resume=resume,
                                                             section_name=resume_model[key]['section_name'])
    return resume_model


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
