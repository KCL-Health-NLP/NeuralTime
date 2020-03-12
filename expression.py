import spacy

class Expression():

    def __init__(self, document, start, end, typ, value = None, start_char = None, end_char = None):

        # start and end relate to tokens indexes, start_char and end_char to character indexes
        # typ is the type : date, time, duration, frequency aor age_related
        # value is the text string relative to the expression


        self.start = start
        self.end = end
        self.start_char = start_char
        self.end_char = end_char

        self.type = typ  # or type ?
        if not value and self.start_char and self.end_char:
            self.value = document.text[self.start_char:self.end_char]
        else:
            self.value = value
        self.document = document

    def set_value(self, value):
        self.value = value


    def get_text(self):
        if self.value:
            return self.value
        return self.document.text[self.start_char:self.end_char]




'''
        if start_char == None:
            self.start_char = document.tokenized_doc[start:start+1].start_char
        else:
            self.start_char = start_char
        if end_char == None:
            self.end_char = document.tokenized_doc[end:end+1].end_char
        else:
            self.end_char = end_char
            '''