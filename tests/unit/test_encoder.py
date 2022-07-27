__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from executor.audioclip_image import AudioCLIPImageEncoder
from jina import Document, DocumentArray, Executor
from PIL import Image


@pytest.fixture(scope="module")
def basic_encoder() -> AudioCLIPImageEncoder:
    return AudioCLIPImageEncoder()


@pytest.fixture(scope="module")
def basic_encoder_no_pre() -> AudioCLIPImageEncoder:
    return AudioCLIPImageEncoder(use_preprocessing=False)


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.batch_size == 32
    assert ex.access_paths == '@r'
    assert ex.use_preprocessing


def test_no_documents(basic_encoder):
    docs = DocumentArray()
    basic_encoder.encode(docs=DocumentArray(), parameters={})
    assert len(docs) == 0


def test_none_docs(basic_encoder):
    basic_encoder.encode(docs=None, parameters={})


def test_docs_no_tensors(basic_encoder):
    docs = DocumentArray([Document()])
    basic_encoder.encode(docs=DocumentArray(), parameters={})
    assert len(docs) == 1
    assert docs[0].embedding is None


def test_err_preprocessing(basic_encoder):
    docs = DocumentArray([Document(tensor=np.ones((3, 100, 100), dtype=np.uint8))])

    with pytest.raises(ValueError, match='If `use_preprocessing=True`'):
        basic_encoder.encode(docs, {})


def test_err_no_preprocessing(basic_encoder_no_pre):
    docs = DocumentArray([Document(tensor=np.ones((100, 100, 3), dtype=np.uint8))])

    with pytest.raises(ValueError, match='If `use_preprocessing=False`'):
        basic_encoder_no_pre.encode(docs, {})


@pytest.mark.gpu
def test_single_image_gpu():
    encoder = AudioCLIPImageEncoder(device='cuda', download_model=True)
    docs = DocumentArray([Document(tensor=np.ones((100, 100, 3), dtype=np.uint8))])
    encoder.encode(docs, {})

    assert docs[0].embedding.shape == (1024,)
    assert docs[0].embedding.dtype == np.float32


def test_single_image(basic_encoder):
    docs = DocumentArray([Document(tensor=np.ones((100, 100, 3), dtype=np.uint8))])
    basic_encoder.encode(docs, {})

    assert docs[0].embedding.shape == (1024,)
    assert docs[0].embedding.dtype == np.float32


def test_single_image_no_preprocessing(basic_encoder_no_pre):
    docs = DocumentArray([Document(tensor=np.ones((3, 224, 224), dtype=np.uint8))])
    basic_encoder_no_pre.encode(docs, {})

    assert docs[0].embedding.shape == (1024,)
    assert docs[0].embedding.dtype == np.float32


def test_batch_different_size(basic_encoder):
    docs = DocumentArray(
        [
            Document(tensor=np.ones((100, 100, 3), dtype=np.uint8)),
            Document(tensor=np.ones((200, 100, 3), dtype=np.uint8)),
        ]
    )
    basic_encoder.encode(docs, {})

    assert docs[0].embedding.shape == (1024,)
    assert docs[0].embedding.dtype == np.float32
    np.testing.assert_allclose(docs[0].embedding, docs[1].embedding)


def test_batch_no_preprocessing(basic_encoder_no_pre):
    docs = DocumentArray(
        [
            Document(tensor=np.ones((3, 224, 224), dtype=np.uint8)),
            Document(tensor=np.ones((3, 224, 224), dtype=np.uint8)),
        ]
    )
    basic_encoder_no_pre.encode(docs, {})

    assert docs[0].embedding.shape == (1024,)
    assert docs[0].embedding.dtype == np.float32
    np.testing.assert_allclose(docs[0].embedding, docs[1].embedding)


@pytest.mark.parametrize(
    "path, expected_counts",
    [
        ['@c', (('@r', 0), ('@c', 3), ('@cc', 0))],
        ['@cc', (('@r', 0), ('@c', 0), ('@cc', 2))],
        ['@r', (('@r', 1), ('@c', 0), ('@cc', 0))],
        ['@cc,r', (('@r', 1), ('@c', 0), ('@cc', 2))],
    ],
)
def test_traversal_path(
    path: str,
    expected_counts: Tuple[str, int],
    basic_encoder: AudioCLIPImageEncoder,
):
    tensor = np.ones((224, 224, 3), dtype=np.uint8)
    docs = DocumentArray([Document(id='root1', tensor=tensor)])
    docs[0].chunks = [
        Document(id='chunk11', tensor=tensor),
        Document(id='chunk12', tensor=tensor),
        Document(id='chunk13', tensor=tensor),
    ]
    docs[0].chunks[0].chunks = [
        Document(id='chunk111', tensor=tensor),
        Document(id='chunk112', tensor=tensor),
    ]
    basic_encoder.encode(docs, parameters={'access_paths': path})
    for path_check, count in expected_counts:
        embeddings = docs[path_check].embeddings
        if count != 0:
            assert len([em for em in embeddings if em is not None]) == count
        else:
            assert embeddings is None


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_size(basic_encoder: AudioCLIPImageEncoder, batch_size: int):
    tensor = np.ones((224, 224, 3), dtype=np.uint8)
    docs = DocumentArray([Document(tensor=tensor) for _ in range(32)])
    basic_encoder.encode(docs, parameters={'batch_size': batch_size})

    for doc in docs:
        assert doc.embedding.shape == (1024,)


def test_embeddings_quality(basic_encoder: AudioCLIPImageEncoder):
    """
    This tests that the embeddings actually "make sense".
    We check this by making sure that the distance between the embeddings
    of two similar images is smaller than everything else.
    """

    data_dir = Path(__file__).parent.parent / 'imgs'
    dog = Document(id='dog', tensor=np.array(Image.open(data_dir / 'dog.jpg')))
    cat = Document(id='cat', tensor=np.array(Image.open(data_dir / 'cat.jpg')))
    airplane = Document(
        id='airplane', tensor=np.array(Image.open(data_dir / 'airplane.jpg'))
    )
    helicopter = Document(
        id='helicopter', tensor=np.array(Image.open(data_dir / 'helicopter.jpg'))
    )

    docs = DocumentArray([dog, cat, airplane, helicopter])
    basic_encoder.encode(docs, {})

    docs.match(docs)
    matches = ['cat', 'dog', 'helicopter', 'airplane']
    for i, doc in enumerate(docs):
        assert doc.matches[1].id == matches[i]
