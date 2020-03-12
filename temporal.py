
class Temporal():
    def __init__(self, type):
        self.type = type





class Time(Temporal):

    def __init__(self, date):
        super().__init__('Time')
        self.date = date

    def print_temporal(self):
        print('TIME')
        print(self.date.token_text)


class Duration(Temporal):

    # start and end are two temporals
    def __init__(self, start, end):
        super().__init__('Duration')
        self.start = start
        self.end = end

    def print_temporal(self):
        print('DURATION')
        self.start.print_temporal()
        print(' --- ')
        self.end.print_temporal()
