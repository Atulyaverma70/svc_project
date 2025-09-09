import os
from pathlib import Path

project_name = "src"

list_of_files = [

    f"{project_name}/__init__.py",

    # Core AI Project Generator logic
    f"{project_name}/generator/__init__.py",
    f"{project_name}/generator/prompt_builder.py",
    f"{project_name}/generator/llm_connector.py",
    f"{project_name}/generator/code_writer.py",
    f"{project_name}/generator/project_builder.py",

    # Project packaging, archiving, exporting
    f"{project_name}/packager/__init__.py",
    f"{project_name}/packager/docker_packager.py",
    f"{project_name}/packager/aws_deployer.py",
    f"{project_name}/packager/github_uploader.py",

    # Input parsing and validation
    f"{project_name}/parser/__init__.py",
    f"{project_name}/parser/synopsis_parser.py",
    f"{project_name}/parser/schema_validator.py",

    # Utilities
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/helpers.py",

    # Logging and exception handling
    f"{project_name}/logger/__init__.py",
    f"{project_name}/exception/__init__.py",

    # Configuration
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/settings.py",
    f"{project_name}/config/prompts.yaml",
    f"{project_name}/config/aws.yaml",

    # APIs (Flask or FastAPI style)
    f"{project_name}/api/__init__.py",
    f"{project_name}/api/routes.py",
    f"{project_name}/api/request_handler.py",

    # Pipelines
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/generate_pipeline.py",
    f"{project_name}/pipeline/deploy_pipeline.py",

    # Entry points
    "app.py",
    "main.py",
    "requirements.txt",
    "Dockerfile",
    ".dockerignore",
    "setup.py",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    ".gitignore",

    # CI/CD and Configs
    ".github/workflows/deploy.yml",
    "config/schema.yaml",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"File already exists at: {filepath}")
