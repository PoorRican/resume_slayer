from setuptools import setup

setup(
    name='slayer',
    version='1.0',
    description='A module for processing resumes and generating summaries',
    author='Your Name',
    author_email='your_email@example.com',
    packages=['slayer'],
    install_requires=[
        'asyncio',
        'langchain',
    ],
)