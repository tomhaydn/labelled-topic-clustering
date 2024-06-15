import os
import codecs
from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.1.0'
DESCRIPTION = 'Super Simple Labelled Topic Clustering'

setup(
      name='labelled-topic-clustering',
      version=VERSION,
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/tomhaydn/labelled-topic-clustering',
      author='Tom Haydn',
      license='MIT',
      package_dir={"": "src"},
      packages=find_packages(where="src"),
      zip_safe=False,
      install_requires=[
            'sentence-transformers==2.2.2',
            'gensim==4.3.0',
            'spacy==3.5.3',
      ],
      python_requires=">=3.9",
      extras_require={
            "dev": [
                  "pytest>=7.0", 
                  "twine>=5.0.1",
                  "pylint>=3.2.3"
            ],
      },
      keywords=[
            "Sentence", "Topic", "Clustering", "Labelling", "Cosine Similarity", 
            "LDA", "HuggingFace", "pyTorch", "Spacy", "NLP", "Deep Learning"
      ],
      classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.9",
            "Operating System :: OS Independent",
            "Development Status :: 5 - Production/Stable",
            "Natural Language :: English"
      ],
)
