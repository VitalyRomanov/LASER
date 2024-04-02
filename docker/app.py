#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from functools import lru_cache

os.environ["PYTHONPATH"] = f"/app/LASER/source:{os.environ.get('PYTHONPATH', '')}"
print(os.environ["PYTHONPATH"])

from flask import Flask, request, jsonify
import os
import socket
import tempfile
from pathlib import Path
import numpy as np
from source.lib.text_processing import Token, BPEfastApply
from source.embed import *

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route("/")
def root():
    print("/")
    html = "<h3>Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>"
    return html.format(name=os.getenv("LASER", "world"), hostname=socket.gethostname())


@app.route("/vectorize")
def vectorize():
    content = request.args.get('q')
    lang = request.args.get('lang')
    embedding = ''
    if lang is None or not lang:
        lang = "en"
    # encoder
    model_dir = Path(__file__).parent / "LASER" / "models"
    encoder_path = model_dir / "bilstm.93langs.2018-12-26.pt"
    bpe_codes_path = model_dir / "93langs.fcodes"
    print(f' - Encoder: loading {encoder_path}')
    encoder = SentenceEncoder(encoder_path,
                              max_sentences=None,
                              max_tokens=12000,
                              sort_kind='mergesort',
                              cpu=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        ifname = tmpdir / "content.txt"
        bpe_fname = tmpdir / 'bpe'
        bpe_oname = tmpdir / 'out.raw'
        with ifname.open("w") as f:
            f.write(content)
        if lang != '--':
            tok_fname = tmpdir / "tok"
            Token(str(ifname),
                  str(tok_fname),
                  lang=lang,
                  romanize=True if lang == 'el' else False,
                  lower_case=True,
                  gzip=False,
                  verbose=True,
                  over_write=False)
            ifname = tok_fname
        BPEfastApply(str(ifname),
                     str(bpe_fname),
                     str(bpe_codes_path),
                     verbose=True, over_write=False)
        ifname = bpe_fname
        EncodeFile(encoder,
                   str(ifname),
                   str(bpe_oname),
                   verbose=True,
                   over_write=False,
                   buffer_size=10000)
        dim = 1024
        X = np.fromfile(str(bpe_oname), dtype=np.float32, count=-1)
        X.resize(X.shape[0] // dim, dim)
        embedding = X
    body = {'content': content, 'embedding': embedding.tolist()}
    return jsonify(body)


@lru_cache()
def load_model(
    encoder: str,
    spm_model: str,
    bpe_codes: str,
    hugging_face=False,
    verbose=False,
    **encoder_kwargs
):
    if spm_model:
        spm_vocab = str(Path(spm_model).with_suffix(".cvocab"))
    else:
        spm_vocab = None
    return SentenceEncoder(
        encoder, spm_vocab=spm_vocab, verbose=verbose, **encoder_kwargs
    )


def select_encoder_and_spm(lang):
    language_model_map = {
        "tt": "laser3-tat_Cyrl.v1",
        "tat": "laser3-tat_Cyrl.v1",
        "tur": "laser3-tur_Latn.v1",
        "kaz": "laser3-kaz_Cyrl.v1",
        "bak": "laser3-bak_Cyrl.v1",
    }

    model_name = language_model_map.get(lang, "laser2")

    assert os.environ.get('LASER'), 'Please set the environment variable LASER'
    LASER = Path(os.environ['LASER'])

    encoder = LASER / "models" / (model_name + ".pt")
    spm = LASER / "models" / (model_name + ".spm")

    return str(encoder), str(spm if spm.is_file() else LASER / "models" / ("laser2" + ".spm"))


@app.route("/vectorize2")
def vectorize2():
    content = request.args.get('q')
    lang = request.args.get('lang')
    embedding = ''
    if lang is None or not lang:
        lang = "en"
    # encoder
    encoder_path, spm_model = select_encoder_and_spm(lang)
    print(f' - Encoder: loading {encoder_path}')
    encoder = load_model(
        encoder_path,
        spm_model=spm_model,
        bpe_codes=None,
        verbose=False
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        ifname = tmpdir / "content.txt"
        bpe_fname = tmpdir / 'bpe'
        bpe_oname = tmpdir / 'out.raw'
        with ifname.open("w") as f:
            f.write(content)
        if lang != '--':
            tok_fname = tmpdir / "tok"
            Token(str(ifname),
                  str(tok_fname),
                  lang=lang,
                  romanize=True if lang == 'el' else False,
                  lower_case=True,
                  gzip=False,
                  verbose=False,
                  over_write=False)
            ifname = tok_fname
        spm_fname = os.path.join(tmpdir, "spm")
        SPMApply(
            str(ifname),
            str(spm_fname),
            str(spm_model),
            lang=lang,
            lower_case=True,
            verbose=False,
            over_write=False,
        )
        ifname = spm_fname
        EncodeFile(encoder,
                   str(ifname),
                   str(bpe_oname),
                   verbose=False,
                   over_write=False,
                   buffer_size=10000)
        dim = 512
        X = np.fromfile(str(bpe_oname), dtype=np.float32, count=-1)
        X.resize(X.shape[0] // dim, dim)
        embedding = X
    body = {'content': content, 'embedding': embedding.tolist()}
    return jsonify(body)


if __name__ == "__main__":
    app.run(debug=True, port=80, host='0.0.0.0')
