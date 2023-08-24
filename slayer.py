from langchain.schema import Document
from typing import List
from history import generate_star_chain
from summary import generate_snippets, generate_summary_chain
from util import cut_sections, chunk_markdown, job_requirement_chain


class Slayer(object):
    summary: str
    experiences: List[Document]
    description: str
    title: str
    response: str

    def __init__(self, resume: str, description: str, title: str):
        super().__init__()

        sections = cut_sections(resume)
        self.summary = sections['summary']
        self.experiences = chunk_markdown(sections['history'])

        self.description = description
        self.title = title

    def process(self) -> None:
        # chains should be executed asynchronously

        requirements_chain = job_requirement_chain()
        requirements = requirements_chain({'desc'})['requirements'].skills

        # handle summary
        snippets = generate_snippets(self.experiences, requirements, self.description)
        overview = generate_summary_chain()({'snippets': snippets, 'desc': self.description})['refined_overview']

        # handle job experience sections

        improve_summary = generate_star_chain()

        experiences = self.experiences.copy()
        for experience in experiences:
            improved = improve_summary({"section": experience.page_content, "requirements": requirements})
            experience.page_content = improved['star']

        print('Overview:\n', overview)
        print('Experiences:\n', experiences)
