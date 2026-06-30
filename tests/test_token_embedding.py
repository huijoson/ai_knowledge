import torch

from src.tokenization import SimpleTokenizer, SimpleEmbedding


def test_simple_tokenizer_encodes_known_tokens_and_decodes_ids():
    tokenizer = SimpleTokenizer(tokens=["我", "喜歡", "學", "AI"])

    ids = tokenizer.encode("我 喜歡 學 AI")

    assert ids == [2, 3, 4, 5]
    assert tokenizer.decode(ids) == ["我", "喜歡", "學", "AI"]
    assert tokenizer.vocab_size == 6


def test_simple_tokenizer_maps_unknown_tokens_to_unk_id():
    tokenizer = SimpleTokenizer(tokens=["我", "喜歡", "學", "AI"])

    ids = tokenizer.encode("我 想 學 AI")

    assert ids == [2, 1, 4, 5]
    assert tokenizer.decode(ids) == ["我", "<unk>", "學", "AI"]


def test_simple_tokenizer_batch_encode_pads_to_longest_sequence():
    tokenizer = SimpleTokenizer(tokens=["我", "喜歡", "學", "AI"])

    batch = tokenizer.batch_encode(["我 喜歡", "我 喜歡 學 AI"])

    assert batch.tolist() == [
        [2, 3, 0, 0],
        [2, 3, 4, 5],
    ]
    assert batch.dtype == torch.long
    assert batch.shape == (2, 4)


def test_simple_embedding_lookup_returns_expected_shape_and_rows():
    embedding = SimpleEmbedding(vocab_size=6, hidden_size=3)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(18, dtype=torch.float32).view(6, 3))

    input_ids = torch.tensor([[2, 3, 0], [4, 5, 1]], dtype=torch.long)
    output = embedding(input_ids)

    assert output.shape == (2, 3, 3)
    assert torch.equal(output[0, 0], torch.tensor([6.0, 7.0, 8.0]))
    assert torch.equal(output[1, 2], torch.tensor([3.0, 4.0, 5.0]))
