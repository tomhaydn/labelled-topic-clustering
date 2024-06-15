from setuptools import find_packages, setup
import os
import codecs

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.0.17'
DESCRIPTION = 'Super Simple Topic Clustering'

setup(
      name='labelled-topic-clustering',
      version=VERSION,
      description='Super Simple Topic Clustering',
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
        "dev": ["pytest>=7.0", "twine>=5.0.1"],
      }
)
