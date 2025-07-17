# [prompt-optimizer-API]

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
## Table of Contents

* [Project Description](#-project-description)
* [Features](#-features)
* [Installation](#-installation)
* [Configuration](#-configuration)
* [Usage](#-usage)
* [Optimization Levels](#-optimization-levels)
* [Understanding `nl_data.py`](#-understanding-nl_datapy)
* [Examples](#-examples)
* [Contributing](#-contributing)
* [License](#-license)
* [Contact](#-contact)
* [Acknowledgements](#-acknowledgements)

## Project Description

The **prompt-optimizer-API** is an API designed to increase efficiency and clarity of prompts sent to LLMs such as GPT, Claude, and DeepSeek. By intelligently removing filler words, redundant phrases, greetings, and performing other linguistic simplifications, it aims to reduce token count, improve prompt comprehension by the LLM, lowering API cost.

This optimizer uses **SpaCy** for robust natural language processing capabilities (like lemmatization, Part-of-Speech tagging, and dependency parsing) and **tiktoken** for accurate token counting(approx. for non-gpt models).

## Features

* **Token Optimization:** Significantly reduces token count for cost-effective LLM interactions.
* **Context Preservation:** Designed to remove unnecessary words while retaining the core intent and meaning of the prompt.
* **Multiple Optimization Levels:** Offers `light`, `medium`, `aggressive`, and `chatbot` predefined optimization strategies.
* **Customizable Optimizations:** Allows users to define custom sets of optimization steps.

## Usage
- NOT FILLED yet

## Config

The core configuration for the `PromptOptimizer` resides in the `app/nl/nl_data.py` file. This file contains various linguistic data structures used by the optimizer, including:

* `general_fillers`: A `set` of common filler words.
* `contextual_fillers`: A `set` of context-dependent filler phrases.
* `greetings`: A `set` of greeting phrases.
* `request_patterns`: A list of `(regex_pattern, replacement_string)` tuples for common request phrases.
* `verbose_replacements`: A dictionary for replacing verbose phrases with concise equivalents.
* `redundant_patterns`: A list of `(regex_pattern, replacement_string)` tuples for redundant pronoun usages.
* `chatbot_optimizations`: A dictionary containing specific optimizations for chatbot prompts, including:
    * `aggressive_fillers` (a `set`)
    * `courtesy_removals` (a `set`)
    * `contractions` (a `dict`)
    * `preposition_simplifications` (a `dict`)
    * `redundant_phrases` (a `dict`)
    * `intensifier_reductions` (a `dict`)
    * `conjunction_simplifications` (a `dict`)
    * `question_simplifications` (a `dict`)
    * `transition_simplifications` (a `dict`)

You can customize these sets, dictionaries, and lists in `nl_data.py` to tailor the optimization behavior to your specific needs.
