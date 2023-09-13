from setuptools import setup

setup(
    name='slayer',
    version='v0.1.2-beta',
    description='A module for processing resumes and generating summaries',
    author='Your Name',
    author_email='your_email@example.com',
    packages=['slayer'],
    install_requires=[
        'asyncio',
        'langchain',
    ],
)