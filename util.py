from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from langchain.text_splitter import MarkdownHeaderTextSplitter
from pydantic import BaseModel
from typing import List


class Skills(BaseModel):
    skills: List[str] = []


def list_job_requirements(description: str) -> List[str]:
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
    chain = LLMChain(prompt=prompt, llm=llm)
    return parser.parse(chain.predict(description=description))


def chunk_resume(resume: str) -> List[Document]:
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


def emulate_section_wording(title: str, desc: str, section: str) -> str:
    """ Improve resume section by copying wording and grammatical syntax between sections

    :param title: job title to emulate
    :param desc: description to emulate
    :param section: resume section to improve

    :return: improved resume section str. Section may be returned have different formatting.
    """
    prompt = PromptTemplate.from_template("""
        You're an expert career consultant with an IQ over 140 working with a special client regarding this job posting.
        Please improve this resume section for this {title} position.
        Improve the section by matching grammatical syntax and lexicon.
        
        This is the job description:\n\n{desc}.
        \n\n
        Here is the resume section:\n{section}
    """)

    grammatical_chain = LLMChain(prompt=prompt, llm=ChatOpenAI(temperature=.85, model_name="gpt-3.5-turbo"))
    return grammatical_chain.predict(title=title, desc=desc, section=section)
