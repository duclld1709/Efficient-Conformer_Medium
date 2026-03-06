import Levenshtein as Lev

class CharErrorRate:

    def __init__(self):
        self.cer = 0.0
        self.n_chars = 0

    def diff_chars(self, transcript, reference):
        self.n_chars = len(reference.replace(' ', ''))
        s1, s2, = transcript.replace(' ', ''), reference.replace(' ', '')
        return float(Lev.distance(s1, s2))

class WordErrorRate:

    def __init__(self):
        self.wer = 0.0
        self.n_tokens = 0

    def diff_words(self, transcript, reference):

        self.n_tokens = len(reference.split())
        
        # build mapping of words to integers
        b = set(transcript.split() + reference.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in transcript.split()]
        w2 = [chr(word2char[w]) for w in reference.split()]
        
        return float(Lev.distance(''.join(w1), ''.join(w2)))