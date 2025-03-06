from setuptools import setup, find_namespace_packages

setup(
    name="thales",
    version="0.1.0",
    description="Integration testing toolkit for Goliath Educational Platform",
    author="Goliath Ed-Tech Team",
    author_email="team@goliath-edu.com",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    install_requires=[
        "pymongo>=4.0.0",
        "neo4j>=5.0.0",
        "qdrant-client>=1.1.1",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "httpx>=0.24.0",
        "pyyaml>=6.0",
        "click>=8.1.3",
        "pydantic>=1.10.7",
        "aiohttp>=3.8.4",
        "python-dateutil>=2.8.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
            "types-PyYAML",
            "types-python-dateutil",
        ],
    },
    entry_points={
        "console_scripts": [
            "thales=thales.cli:main",
        ],
    },
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)