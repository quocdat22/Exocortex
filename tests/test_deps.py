def test_evaluation_dependencies_importable():
    import langchain
    import pandas
    import ragas

    assert ragas is not None
    assert langchain is not None
    assert pandas is not None
