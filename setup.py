from setuptools import setup, find_packages

setup(
    name="aigie",
    version="0.1.0",
    description="The Package that fixes you AI world",
    author="AG",
    author_email="support@aigie.ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tenacity",
        "langchain-core", 
        "langgraph",
        "google-cloud-logging"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
