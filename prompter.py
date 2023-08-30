"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union

def neither_is_none_or_both_are_none(a, b):
    if a is None and b is None:
        return True
    if a is not None and b is not None:
        return True
    return False

def select_first_non_none(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_path: str = None, template_object=None, verbose: bool = False):
        if neither_is_none_or_both_are_none(template_path, template_object):
            raise ValueError("Either template_path or template_object must be provided")

        self._verbose = verbose
        if template_path is not None:
            if not osp.exists(template_path):
                raise ValueError(f"Can't read {template_path}")
            with open(template_path) as fp:
                self.template = json.load(fp)
            if self._verbose:
                print(
                    f"Using prompt template {template_path}: {self.template['description']}"
                )
        else:
            self.template = template_object

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
