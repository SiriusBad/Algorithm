import time

# 自定义时间记录hook
class TimeHook:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        # 记录算法开始时间
        self.start_time = time.time()

    def end(self):
        # 记录算法结束时间
        self.end_time = time.time()

    def execution_time(self):
        # 计算算法的执行时间
        return self.end_time - self.start_time

# 示例用法
if __name__ == "__main__":
    # 创建TimeHook对象
    time_hook = TimeHook()

    # 模拟算法执行
    time_hook.start()
    # 执行你的算法代码
    time.sleep(2)  # 示例中的模拟算法代码，实际应替换为你的算法
    time_hook.end()

    # 获取算法执行时间
    execution_time = time_hook.execution_time()
    print("Algorithm execution time:", execution_time, "seconds")
