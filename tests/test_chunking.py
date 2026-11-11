from app.main import split_into_chunks

def test_split_into_chunks_returns_content():
    chunks = split_into_chunks("one two three four five", words_per_chunk=3, overlap=1)
    assert chunks
    assert chunks[0] == "one two three"
