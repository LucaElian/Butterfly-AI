from butterfly.tokenizer import ButterflyTokenizer

def test_tokenizer_roundtrip(tmp_path):
    p=tmp_path/"tiny.txt";p.write_text("Hola Butterfly. ¿Cómo estás? Butterfly aprende palabras y subpalabras. "*80,encoding="utf-8")
    tok=ButterflyTokenizer.train_subword([p],vocab_size=400)
    text="Hola, Butterfly! ¿Cómo estás? 🦋"
    assert tok.decode(tok.encode(text))==text
    assert tok.vocab_size>=259
