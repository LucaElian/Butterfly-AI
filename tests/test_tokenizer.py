from butterfly.tokenizer import ButterflyTokenizer

def test_tokenizer_roundtrip():
    tok=ButterflyTokenizer.train("Hola Butterfly. Hola mundo. Butterfly aprende palabras. "*10,target_vocab=320)
    text="Hola, Butterfly! ¿Cómo estás? 🦋"
    assert tok.decode(tok.encode(text))==text
    assert tok.vocab_size>=259
