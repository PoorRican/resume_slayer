from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


class Slayer(object):
    resume: str
    description: str
    title: str
    response: str

    def __init__(self, resume: str, description: str, title: str):
        super().__init__()
        self.resume = resume
        self.description = description
        self.title = title

    def predict(self) -> str:
        chat_model = ChatOpenAI(temperature=.75, model_name="gpt-3.5-turbo")
        system_message = SystemMessagePromptTemplate.from_template(
            "You're an expert career consultant with an IQ over 140. Please rewrite my resume to exude"
            "competence and clearly demonstrate how and why I am a perfect fit for the {job_title} position by"
            "reusing keywords, phrases, and skills, and experience. Do not add anything that is not there, but expand"
            "on anything that is in my resume. This is my current resume:\n{resume}")
        job_desc_prompt = HumanMessagePromptTemplate.from_template("Here is the job description:\n{job_desc}")
        chat_prompt = ChatPromptTemplate.from_messages([system_message, job_desc_prompt])

        messages = chat_prompt.format_prompt(resume=self.resume,
                                             job_desc=self.description,
                                             job_title=self.title).to_messages()
        response = chat_model.predict_messages(messages)
        self.response = str(response)
        return self.response
