def test_evaluation_dependencies_importable():
    import langchain
    import langchain_community
    import langchain_openai
    import pandas
    import ragas
    import tabulate

    assert ragas is not None
    assert langchain is not None
    assert pandas is not None
