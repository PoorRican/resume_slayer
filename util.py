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


