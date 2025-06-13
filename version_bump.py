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
"""This Python script will bump the version of the package
based on the release type passed as an argument in the workflow"""
import sys
import re

#Read the current version from pyproject.toml
with open("pyproject.toml", "r", encoding="utf-8") as f:
    toml_file = f.read()
current_version = re.search(r'version = "(\d+\.\d+\.\d+)"', toml_file).group(1)

#Split version into components(MAJOR.MINOR.PATCH)
MAJOR, MINOR, PATCH = current_version.split(".")

#version type is the argument passed in the shell command
release_type = sys.argv[1]

#increment the version based on release type
if release_type == "Major":
    MAJOR = int(MAJOR) + 1
    MINOR = 0
    PATCH = 0
elif release_type == "Minor":
    MINOR = int(MINOR) + 1
    PATCH = 0
elif release_type == "Patch":
    PATCH = int(PATCH) + 1
else:
    print("Invalid release type")
    sys.exit(1)

#Write the updated version to pyproject.toml
new_version = f"{MAJOR}.{MINOR}.{PATCH}"
with open("pyproject.toml", "w",encoding="utf-8") as f:
    f.write(re.sub(r'version = "\d+\.\d+\.\d+"', f'version = "{new_version}"', toml_file))
print(f"version updated to {new_version}")
