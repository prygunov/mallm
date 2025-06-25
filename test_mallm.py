from langgraph_agent import run_query
import asyncio

def test_chorus():
    query = """What is the last word before the second chorus of the King of Pop's fifth single from his sixth studio album?"""
    result = asyncio.run(run_query(query))
    assert 'stare' in result

def test_math():
    query = "What is 2 + 2?"
    result = asyncio.run(run_query(query))
    assert '4' in result