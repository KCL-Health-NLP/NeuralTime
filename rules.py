from spacy.matcher import Matcher
from expression import Expression

class Rule():

    def __init__(self, type):
        self.type = type


class TokenRule(Rule):

    def __init__(self, name, pattern, nlp, result = None):
        super().__init__('Token')
        self.name = name
        self.pattern = pattern
        self.result = result
        self.matcher = Matcher(nlp.vocab)
        self.matcher.add(self.name,  result, self.pattern)

    def apply(self, input_tokens):
        matches = self.matcher(input_tokens)  #input tokens has to be a text tokenized by spacy model
        # extracting the expressions based on the matches
        print(matches)
        expressions = [Expression(start, end, self) for id, start, end in matches]
        return expressions


class TemporalRule(Rule):

    def __init__(self, result=lambda x: None):
        super().__init__('Temporal')
        self.result = result

    def apply(self, inputs):
        return [self.result(i) for i in inputs]


