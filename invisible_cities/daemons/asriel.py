from . daemon import Daemon

class Asriel(Daemon):

    def __init__(self):
        print('I am Asriel')

    def run(self):
        print('Asriel runs')

    def end(self):
        print('Asriel ends')
