from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

def porterStemmer(tokens):
    porter = PorterStemmer()
    stem_tokens = [porter.stem(w) for w in tokens]
    return stem_tokens

def snowBallStemmer(tokens):
    snowball = SnowballStemmer('english')
    stem_tokens = [snowball.stem(w) for w in tokens]
    return stem_tokens

def lancasterStemmer(tokens):
    lancaster = LancasterStemmer()
    stem_tokens = [lancaster.stem(w) for w in tokens]
    return stem_tokens