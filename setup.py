from setuptools import setup, find_packages
from typing import List

HYPEN = '-e .'

def get_requirements(file_path:str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN in requirements:
            requirements.remove(HYPEN)

    return requirements

setup(
    name = "Heart Failure",
    version= "0.0.1",
    author = "Ansh Patel",
    author_email = "ansh_patel@student.uml.edu",
    packages = find_packages,
    install_requires = get_requirements("requirements.txt")
)