from collections import defaultdict
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
import itertools
import nltk
nltk.download('punkt')

# Combine training set from two sources so that we have slightly more varied dictionary
train1 = open('train/news-commentary-v12.tsv', 'r', encoding='utf-8')
train2 = open('train/europarl-v7.tsv', 'r', encoding='utf-8')

znaki_interpunkcyjne = ":;.,?!/"

dictionary = []
cz = []
eng = []
for x in train1.readlines()[33000:118000]:
        x = x.replace("The ", "")
        x = x.lower()
        x = x.replace("\n","")
        x = x.replace(" the ", " ")
        x = x.replace(" of ", " ")
        x = re.sub('[' + znaki_interpunkcyjne + ']', '', x)
        x = x.split("\t")
        try:
            dictionary.append(x)
            eng.append(x[1])
            cz.append(x[0])
        except IndexError:
            pass
for x in train2.readlines()[33000:74300]:
        x = x.replace("The ", "")
        x = x.lower()
        x = x.replace("\n","")
        x = x.replace(" the ", " ")
        x = x.replace(" of ", " ")
        x = re.sub('[' + znaki_interpunkcyjne + ']', '', x)
        x = x.split("\t")
        try:
            dictionary.append(x)
            eng.append(x[1])
            cz.append(x[0])
        except IndexError:
            pass
for x in train2.readlines()[123000:173000]:
        x = x.replace("The ", "")
        x = x.lower()
        x = x.replace("\n","")
        x = x.replace(" the ", " ")
        x = x.replace(" of ", " ")
        x = re.sub('[' + znaki_interpunkcyjne + ']', '', x)
        x = x.split("\t")
        try:
            dictionary.append(x)
            eng.append(x[1])
            cz.append(x[0])
        except IndexError:
            pass
dct = dict(zip(cz, eng))

# Initialize values uniformly 
t = defaultdict(int)
n = defaultdict(int)
stotal = defaultdict(int)
for key in dct:
    eng_sentence = nltk.word_tokenize(dct[key])
    cz_sentence = nltk.word_tokenize(key)
    for e in eng_sentence:
        for cz in cz_sentence:
            n[e] += 1
            
for key in dct:
    eng_sentence = nltk.word_tokenize(dct[key])
    cz_sentence = nltk.word_tokenize(key)
    for e in eng_sentence:
        for cz in cz_sentence:
            t[e, cz] = 1/n[e]

# IBM Model 1 algorithm - pseudocode taken from lecture materials
i = 0
for _ in itertools.repeat(None,25):
            count = defaultdict(int)
            total = defaultdict(int)
            for key in dct:
                eng_sentence = nltk.word_tokenize(dct[key])
                cz_sentence = nltk.word_tokenize(key)
                for e in eng_sentence:
                    stotal[e] = 0
                    for cz in cz_sentence:
                        stotal[e] += t[e, cz]
                for e in eng_sentence:
                    for cz in cz_sentence:
                        count[e, cz] += t[e, cz] / stotal[e]
                        total[cz] += t[e, cz] / stotal[e]
            for cz in cz_sentence:
                for e in eng_sentence:
                    t[e, cz] = count[e, cz] / total[cz]

# best translations -- if not found, we do not alter the word
for cz in cz_sentence:
    for e in eng_sentence:
        if t[e, cz] <= 0.001:
            t.pop[e, cz]
    
def find_word_translation(czech_word):
    best_value = czech_word
    best_score = 0
    for key in t:
        if czech_word == key[1]:
            if t[key] > best_score:
                best_value = key[0]
                best_score = t[key]
    return(best_value)

test = open('test-A/in.tsv', 'r', encoding='utf-8')
with open('test-A/out.tsv', 'w', encoding='utf-8') as test_results:
    test_sentences = []
    translations = {}
    for x in test.readlines():
        x = x.lower()
        x = x.replace("\n","")
        x = re.sub('[' + znaki_interpunkcyjne + ']', '', x)
        x = x.split("\t")
        test_sentences.append(nltk.word_tokenize(x[0]))  
        for lines in test_sentences:
            translation = []
            for a in lines:
                if a not in translations:
                    translations[a] = find_word_translation(a)
                    translation.append(translations[a])
                else:
                    a = translations[a]
                    translation.append(a)
        translation = (TreebankWordDetokenizer().detokenize(translation) + '.\n')
        test_results.write(translation.capitalize())     

dev = open('dev-0/in.tsv', 'r', encoding='utf-8')
with open('dev-0/out.tsv', 'w', encoding='utf-8') as dev_results:
    dev_sentences = []
    translations = {}
    for x in dev.readlines():
        x = x.lower()
        x = x.replace("\n","")
        x = re.sub('[' + znaki_interpunkcyjne + ']', '', x)
        x = x.split("\t")
        dev_sentences.append(nltk.word_tokenize(x[0]))  
        for lines in dev_sentences:
            translation = []
            for a in lines:
                if a not in translations:
                    translations[a] = find_word_translation(a)
                    translation.append(translations[a])
                else:
                    a = translations[a]
                    translation.append(a)
        translation = (TreebankWordDetokenizer().detokenize(translation) + '.\n')
        dev_results.write(translation.capitalize())     
