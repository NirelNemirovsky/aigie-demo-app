from setuptools import setup, find_packages

setup(
    name="aigie",
    version="0.2.0",
    description="The Package that fixes your AI world with Trail Taxonomy error classification and Gemini AI remediation",
    author="AG",
    author_email="support@aigie.ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tenacity",
        "langchain-core", 
        "langgraph",
        "google-cloud-logging",
        "google-cloud-aiplatform"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="ai error handling langgraph gemini trail-taxonomy remediation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
