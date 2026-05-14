class Board:
    def __init__(self):
        self.data = None

    def set_data(title,writer):
        self.title = title
        self.writer = writer
        self.cnt=0
    def cnt_plus(self):
        self.cnt+=1

board1 = Board()
board2 = Board()
board1.set_data("제목1","작성자1")
board2.set_data("제목2","작성자2")