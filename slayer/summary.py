# These chains are focused on improving the summary/overview section
import asyncio

from pydantic import BaseModel
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import SequentialChain
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import Document
from typing import List, Set, Union

from .util import relevant_skills_chain


class Stories(BaseModel):
    """ Class to store a list of stories as strings.

    Parsing using `pydantic` has been implemented to avoid incidentally splitting strings that contain commas
    """
    stories: List[str] = []


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


def extract_stories_chain() -> LLMChain:
    parser = PydanticOutputParser(pydantic_object=Stories)
    _format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate(
        template="""
        Extract {artifacts} from the following resume experience section that demonstrate {attribute}.
        
        Respond with 'None' for Python if there are no relevant {artifacts} to {attribute}.
        
        Here is the resume section (surrounded by ``):
        
        `{section}`
        
        {format_instructions}
        """,
        input_variables=["attribute", "section"],
        partial_variables={
            'format_instructions': _format_instructions,
            'artifacts': 'stories',
        })

    return LLMChain(llm=ChatOpenAI(temperature=.2, model_name="gpt-3.5-turbo"),
                    output_key='stories',
                    prompt=prompt,
                    output_parser=parser)


def extract_three_things_chain() -> LLMChain:
    """ Accepts job experience, returns list of three things recruiter should know about this job experience. """
    parser = PydanticOutputParser(pydantic_object=Stories)
    format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate(template="""
    Extract 3 distinguishing statements from the following job experience summary that
    make this candidate stand out from the crowd to a job recruiter.
    
    Statements should express a whole idea while remaining concise.

    Here is the job experience summary (surrounded by ``):
    `{section}`

    {format_instructions}
    """,
                            input_variables=["section"],
                            partial_variables={'format_instructions': format_instructions}
                            )

    llm = ChatOpenAI(temperature=.4, model_name='gpt-3.5-turbo')
    return LLMChain(llm=llm, output_parser=parser, prompt=prompt, output_key='three_things')


async def generate_snippets(experiences: List[Union[Document, str]], skills: List[str], description: str) -> Set[str]:
    """ Extract stories and statements from a list of job experiences """
    three_things_chain = extract_three_things_chain()
    _relevant_skills_chain = relevant_skills_chain()
    stories_chain = extract_stories_chain()

    async def _handle_three_things() -> List[str]:
        tasks = await asyncio.gather(*[three_things_chain.arun(i) for i in experiences])
        stories = []
        [stories.extend(i.stories) for i in tasks]
        return stories

    async def _handle_stories() -> List[str]:
        relevant_skills = await asyncio.gather(*[
            _relevant_skills_chain.arun({'section': i,
                                         'requirements': skills}) for i in experiences])

        tasks = []
        for _experience, experience_skills in zip(experiences, relevant_skills):
            for skill in experience_skills:
                task = stories_chain.arun({'section': _experience, 'desc': description, 'attribute': skill})
                tasks.append(task)
        stories = await asyncio.gather(*tasks)
        sub_stories = []
        for i in stories:
            sub_stories.extend(i.stories)
        return sub_stories

    extracted_stories, extracted_three_things = await asyncio.gather(
        _handle_stories(),
        _handle_three_things()
    )
    return {*extracted_stories, *extracted_three_things}


def generate_summary_chain() -> SequentialChain:
    """ Chain that generates resume summary from candidate clauses """

    llm = ChatOpenAI(temperature=0.4, model_name="gpt-3.5-turbo")

    # Narrow down candidate clauses
    top_snippets_prompt = PromptTemplate.from_template("""
    We will be filtering a list of bullet points.
    Out of the bullets, select the top 10 that will make the candidate stand out for the given job description.

    The returned bullets should reflect the original content, and not be an excerpt from the job description.
    Also, Return the top snippets as bullet points.

    This is the job description (surrounded by ``):
        `{desc}`

    This is the list of candidate snippets (surrounded by ``):
    `{snippets}`
    """)
    top_snippets_chain = LLMChain(prompt=top_snippets_prompt, llm=llm, output_key='refined_snippets')

    summary_prompt = PromptTemplate.from_template("""
    Using the given snippets, generate a brief resume summary as bullet points.

    Here is a list of snippets (surrounded by ``):
    `{refined_snippets}`
    """)
    summary_chain = LLMChain(prompt=summary_prompt, llm=llm, output_key='summary_overview')

    refine_prompt = PromptTemplate.from_template("""
    Refine the given list of bullet points to 3-5 so that it stands out to someone reading the given job
    description.

    Here is a list of bullet points (surrounded by ``):
    `{summary_overview}`

    Here is the job description (surrounded by ``):
    `{desc}`
    """)
    refine_chain = LLMChain(prompt=refine_prompt, llm=llm, output_key='refined_overview')

    return SequentialChain(chains=[top_snippets_chain, summary_chain, refine_chain],
                           input_variables=['desc', 'snippets'],
                           output_variables=['refined_overview'])
