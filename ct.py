class A():

    def __init__(self):
        print('init A')
        self.method_1

    def method_1(self):
        print('method 1 of A')
        self.method_2()

    def method_2(self):
        print('method 2 of A')

class B(A):
    
    def __init__(self):
        # super(B, self). __init__()
        # print('init B')
        self.method_1()

    # def method_1(self):
    #     print('method 1 of A')
    #     self.method_2

    def method_2(self):
        print('method 2 of B')

b = B()