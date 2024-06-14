from setuptools import find_packages, setup

with open("app/README.md", "r") as f:
      long_description = f.read()

setup(
      name='labelled-topic-clustering',
      version='1.0.10',
      description='Super Simple Topic Clustering',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/tomhaydn/super-simple-topic-clustering',
      author='Tom Haydn',
      license='MIT',
      package_dir={"": "app"},
      packages=find_packages(where="app"),
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
