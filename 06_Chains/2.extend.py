class Test(object):
    def __init__(self, name):
        self.name = name

    def __or__(self, other):
        return MySequence(self, other)

    def __str__(self):
        return self.name


class MySequence(object):
    def __init__(self, *args):
        self.sequence = []
        for arg in args:
            self.sequence.append(arg)

    def __or__(self, other):
        self.sequence.append(other)
        return self

    def run(self):
        for i in self.sequence:
            print(i)


if __name__ == "__main__":
    # 测试一下这个链式调用
    t1 = Test("第一步：处理提示词")
    t2 = Test("第二步：调用大模型")
    t3 = Test("第三步：解析输出结果")
    
    # 使用 | 符号将它们连接起来，这就是 LCEL (LangChain Expression Language) 的底层原理
    chain = t1 | t2 | t3
    
    print("开始运行 Chain:")
    chain.run()
