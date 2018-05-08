"""

"""
import re, string
import unicodedata


def cleanline(txt):
    return txt.strip('\n')


def to_lowercase(text):
    return text.lower().strip()


def remove_all_punctuations(text):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    text = regex.sub(' ', text).strip()
    return " ".join(text.split()).strip()


def remove_basic_punctuations(text):
    text = text.replace('.','')
    text = text.replace(',','')
    text = text.replace('?','')
    text = text.replace('!','')
    text = text.replace(';','')
    text = text.replace('-',' ')
    return text.strip()


def remove_spaced_single_punctuations(text):
    wds = text.split()
    return " ".join([w for w in wds if len(w)>1 or re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', w).strip() != '']).strip()


def space_out_punctuations(text):
    return re.sub(r"([\w\s]+|[^\w\s]+)\s*", r"\1 ", text).strip()


def remove_numbers(text):
    return re.sub(r' \d+ ',' ', text).strip()


def replace_numbers(text):
    return re.sub(r' \d+ ',' *#NUMBER#* ', text).strip()


def replace_accents(text):
    text = text.decode('utf-8')
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
    text = text.replace('-LRB-','(')
    text = text.replace('-RRB-',')')
    return text.strip()