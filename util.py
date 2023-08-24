from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser
from langchain.chains import LLMChain
from langchain.text_splitter import MarkdownHeaderTextSplitter
from pydantic import BaseModel
from typing import List


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
        Here is the job description:\n{desc}
    """,
                            input_variables=["desc"],
                            partial_variables={'format_instructions': format_instructions})
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
    return LLMChain(prompt=prompt, llm=llm, output_parser=parser, output_key='requirements')


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

    section_names = ('resume summary', 'job history / experience')
    resume_model = {'summary': '',
                    'history': ''}
    classify_chain = LLMChain(prompt=prompt, llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
    for name, key in zip(section_names, resume_model.keys()):
        resume_model[key] = classify_chain.predict(resume=resume,
                                                   section_name=name)
    return resume_model


def relevant_skills_chain() -> LLMChain:
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


def structure_plaintext_chain() -> LLMChain:
    """ Add a resume markdown structure to plaintext """
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

    prompt = PromptTemplate.from_template("""
    You're an expert career consultant with an IQ over 140 working with a special client regarding this job posting.
    Add markdown structure to this plaintext resume.
    
    This is the resume:
    {section}
    """)
    return LLMChain(prompt=prompt, llm=llm)


def format_resume_chain() -> LLMChain:
    """ Format the given text as a resume """
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

    prompt = PromptTemplate.from_template("""
    You're an expert career consultant with an IQ over 140 working with a special client regarding this job posting.
    Format this resume text into a proper markdown document.
    
    This is the resume text:
    {section}
    """)
    return LLMChain(prompt=prompt, llm=llm)
