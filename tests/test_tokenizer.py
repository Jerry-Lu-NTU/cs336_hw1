from __future__ import annotations

import json
import os
import resource
import sys

import psutil
import pytest
import tiktoken

from .adapters import get_tokenizer
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode

# 中文导读：
# 这个文件验证 tokenizer 的 encode/decode 行为。
# 需要实现接口：get_tokenizer(vocab, merges, special_tokens)。
# get_tokenizer 应返回一个对象，至少提供：
# - encode(text: str) -> list[int]
# - decode(ids: list[int]) -> str
# - encode_iterable(iterable: Iterable[str]) -> Iterable[int]
#
# 测试会使用 GPT-2 的 vocab/merges fixture 构造你的 tokenizer，
# 并与 tiktoken 的 GPT-2 tokenizer 对比。因此 bytes 处理、pre-tokenization、
# merge 顺序、特殊 token 和 unicode roundtrip 都要严格处理。

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"


# 测试辅助 decorator：限制某个函数可额外使用的内存。
# 主要用于检查 encode_iterable 是否真的流式处理，而不是一次性读完整文件。
def memory_limit(max_mem):
    def decorator(f):
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            prev_limits = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (process.memory_info().rss + max_mem, -1))
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                # Even if the function above fails (e.g., it exceeds the
                # memory limit), reset the memory limit back to the
                # previous limit so other tests aren't affected.
                resource.setrlimit(resource.RLIMIT_AS, prev_limits)

        return wrapper

    return decorator


# 测试辅助函数：从 GPT-2 fixture 文件读取 vocab/merges，并转换为作业接口要求的 bytes 形式。
# 你不需要实现这个函数；你需要实现的是它最后调用的 get_tokenizer(...)。
def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return get_tokenizer(vocab, merges, special_tokens)


# 测试目标：空字符串 encode 后能 decode 回空字符串。
# 覆盖边界：encode("") 应返回空 id 列表，而不是报错或返回特殊 token。
def test_roundtrip_empty():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = ""
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


# 测试目标：空字符串的 token id 序列与 tiktoken GPT-2 一致。
def test_empty_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = ""

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == []

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


# 测试目标：单个 ASCII 字符能 roundtrip。
def test_roundtrip_single_character():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "s"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


# 测试目标：单个 ASCII 字符的 token id 与 tiktoken 一致。
def test_single_character_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "s"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == ["s"]

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


# 测试目标：单个 unicode 字符能 roundtrip。
# 这要求 tokenizer 内部正确处理 UTF-8 bytes，而不是按 Python 字符粗暴切分。
def test_roundtrip_single_unicode_character():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "🙃"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


# 测试目标：单个 unicode 字符的 token id 与 tiktoken 一致。
def test_single_unicode_character_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "🙃"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


# 测试目标：普通英文句子能 roundtrip。
def test_roundtrip_ascii_string():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "Hello, how are you?"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


# 测试目标：普通英文句子的分词片段符合 GPT-2 风格。
# 注意 `" how"` 这类带前导空格的 token 是正常现象，说明空格参与了 bytes/token 表示。
def test_ascii_string_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Hello, how are you?"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    # assert ids == reference_ids

    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string == ["Hello", ",", " how", " are", " you", "?"]

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


# 测试目标：包含重音字符和 emoji 的字符串能 roundtrip。
def test_roundtrip_unicode_string():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    test_string = "Héllò hôw are ü? 🙃"
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


# 测试目标：包含多字节 unicode 字符的字符串，token id 与 tiktoken 一致。
def test_unicode_string_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw are ü? 🙃"

    reference_ids = reference_tokenizer.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


# 测试目标：含特殊 token 的 unicode 字符串能 roundtrip，且特殊 token 不被拆开。
def test_roundtrip_unicode_string_with_special_tokens():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    encoded_ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    # Ensure the special <|endoftext|> token is preserved
    assert tokenized_string.count("<|endoftext|>") == 3

    decoded_string = tokenizer.decode(encoded_ids)
    assert test_string == decoded_string


# 测试目标：允许特殊 token 时，结果与 tiktoken 的 allowed_special 行为一致。
def test_unicode_string_with_special_tokens_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"

    reference_ids = reference_tokenizer.encode(test_string, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == test_string
    assert reference_tokenizer.decode(reference_ids) == test_string


# 测试目标：特殊 token 有重叠时，长特殊 token 应作为一个整体被保留。
# 例如 "<|endoftext|><|endoftext|>" 不能先被拆成两个短特殊 token。
def test_overlapping_special_tokens():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )
    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"

    ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    # Ensure the double <|endoftext|><|endoftext|> is preserved as a single token
    assert tokenized_string.count("<|endoftext|>") == 1
    assert tokenized_string.count("<|endoftext|><|endoftext|>") == 1
    # Test roundtrip
    assert tokenizer.decode(ids) == test_string


# 测试目标：真实 fixture 文本 address.txt 能 encode/decode roundtrip。
def test_address_roundtrip():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "address.txt") as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


# 测试目标：address.txt 的 token id 与 tiktoken GPT-2 一致。
def test_address_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    corpus_path = FIXTURES_PATH / "address.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents)
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


# 测试目标：德语文本能 roundtrip，覆盖非英语字符和标点。
def test_german_roundtrip():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "german.txt") as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


# 测试目标：德语文本 token id 与 tiktoken GPT-2 一致。
def test_german_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    corpus_path = FIXTURES_PATH / "german.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents)
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


# 测试目标：TinyStories 样本文本能 roundtrip，覆盖较长文本。
def test_tinystories_sample_roundtrip():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
        corpus_contents = f.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


# 测试目标：TinyStories 样本文本 token id 与 tiktoken 一致，且允许 <|endoftext|> 特殊 token。
def test_tinystories_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "tinystories_sample.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


# 测试目标：特殊 token 后面跟换行时，encode 行为仍与 tiktoken 一致。
# 这是 pre-tokenization 和特殊 token 切分的常见边界条件。
def test_encode_special_token_trailing_newlines():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "special_token_trailing_newlines.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


# 测试目标：特殊 token 与双换行、非空白字符相邻时，encode 行为仍与 tiktoken 一致。
def test_encode_special_token_double_newline_non_whitespace():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "special_token_double_newlines_non_whitespace.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids

    assert tokenizer.decode(ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


# 测试目标：encode_iterable 可以逐块/逐行读取文本并保持 roundtrip。
# 这要求流式 encode 的边界处理不能丢字符，也不能改变 token 顺序。
def test_encode_iterable_tinystories_sample_roundtrip():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    all_ids = []
    with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
        corpus_contents = f.read()
    assert tokenizer.decode(all_ids) == corpus_contents


# 测试目标：encode_iterable 的输出 token id 与一次性 encode/tiktoken 结果一致。
def test_encode_iterable_tinystories_matches_tiktoken():
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    corpus_path = FIXTURES_PATH / "tinystories_sample.txt"
    with open(corpus_path) as f:
        corpus_contents = f.read()
    reference_ids = reference_tokenizer.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    all_ids = []
    with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    assert all_ids == reference_ids

    assert tokenizer.decode(all_ids) == corpus_contents
    assert reference_tokenizer.decode(reference_ids) == corpus_contents


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="rlimit support for non-linux systems is spotty.",
)
# 测试目标：encode_iterable 在较大文件上不应额外占用太多内存。
# Linux 下会限制额外内存为 1MB；因此实现应流式产出 id。
def test_encode_iterable_memory_usage():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        ids = []
        for _id in _encode_iterable(tokenizer, f):
            ids.append(_id)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="rlimit support for non-linux systems is spotty.",
)
@pytest.mark.xfail(reason="Tokenizer.encode is expected to take more memory than allotted (1MB).")
# 这个测试预期失败：一次性 encode 大文本可以比 1MB 多用内存。
# 它存在的目的是和上面的 encode_iterable 形成对比。
def test_encode_memory_usage():
    """
    We expect this test to fail, since Tokenizer.encode is not expected to be memory efficient.
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        contents = f.read()
        _ = _encode(tokenizer, contents)


# 内存限制包装：只限制 encode_iterable 调用本身，避免影响其他测试。
@memory_limit(int(1e6))
def _encode_iterable(tokenizer, iterable):
    """
    We place tokenizer.encode_iterable into a separate function so we can limit memory
    for just this function. We set the memory limit to 1MB.
    """
    yield from tokenizer.encode_iterable(iterable)


# 内存限制包装：只限制普通 encode 调用本身。
@memory_limit(int(1e6))
def _encode(tokenizer, text):
    """
    We place tokenizer.encode into a separate function so we can limit memory
    for just this function. We set the memory limit to 1MB.
    """
    return tokenizer.encode(text)
