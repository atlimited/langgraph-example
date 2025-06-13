#!/bin/bash
export TAVILY_API_KEY=test_key
export OPENROUTER_API_KEY=test_key  
export ANTHROPIC_API_KEY=test_key
export OPENAI_API_KEY=test_key
python -m pytest tests/ -v