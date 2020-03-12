
import spacy

## Regular Expressions

import re

test_string = 'This string is a test.'
pattern = '\ws'
a = re.finditer(pattern, test_string)


for match in a:
    print(match)


### Import a file

f = open('data/pediatrics/207-pediatric-letter.txt',"r")
string = f.read()
#print(string)


nlp = spacy.load('en_core_web_md')
document = nlp(string)

#for token in document:
    #print(token.text, token.lemma_, token.pos_ )
    #print(token.has_vector)

string = ('apple orange car')
doc = nlp(string)
a, o, c = doc

print(a.similarity(o))
print(a.similarity(c))

