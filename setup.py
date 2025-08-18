from setuptools import setup, find_packages

setup(
    name="aigie_demo_app",
    version="0.1.0",
    description="The Package that fixes you AI world",
    author="AG",
    author_email="support@aigie.ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["tenacity", "langchain-core"],
)
