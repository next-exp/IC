from . daemon import Daemon

class Lyra(Daemon):

    def __init__(self):
        print('I am Lyra')

    def run(self):
        print('Lyra runs')

    def end(self):
        print('Lyra ends')
