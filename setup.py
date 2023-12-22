# Copyright 2023 American Express Travel Related Services Company, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
"""setup file to build the package"""
from setuptools import setup
from os import path

with open('requirements.txt', encoding="utf-8") as f:
    requirements = f.read().splitlines()

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='connectchain',
    version='0.0.1',
    author='American Express',
    url='https://github.com/americanexpress/connectchain',
    description='Enterprise-focused Generative AI Toolset built upon Langchain',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['langchain', 'connectchain', 'enterprise'],
    license='Apache 2.0 license',
    python_requires='>=3.11',
    packages=['connectchain',
              'connectchain.tools',
              'connectchain.utils',
              'connectchain.prompts',
              'connectchain.chains',
              'connectchain.lcel',
              'connectchain.orchestrators'],
    install_requires=requirements
)
