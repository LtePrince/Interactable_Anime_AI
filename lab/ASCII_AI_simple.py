import numpy as np
import time
import random
import threading
import queue
from collections import deque

class DotMatrixAgent:
    def __init__(self, width=8, height=8):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.position = [width//2, height//2]
        self.memory = deque(maxlen=10)
        
        # 情绪系统
        self.emotion_value = 5.0  # 情绪值 (1-10)
        self.last_input_time = time.time()
        self.input_queue = queue.Queue()
        self.running = True
        
        # 定义基本符号 - 根据情绪状态
        self.symbols = {
            'sad': [  # 情绪值 1-3
                [0,0,0,0,0],
                [0,1,0,1,0],
                [0,0,0,0,0],
                [0,1,1,1,0],
                [1,0,0,0,1]
            ],
            'neutral': [  # 情绪值 4-6
                [0,0,0,0,0],
                [0,1,0,1,0],
                [0,0,0,0,0],
                [0,0,0,0,0],
                [1,1,1,1,1]
            ],
            'happy': [  # 情绪值 7-10
                [0,0,0,0,0],
                [0,1,0,1,0],
                [0,0,0,0,0],
                [1,0,0,0,1],
                [0,1,1,1,0]
            ]
        }
        self.current_symbol = 'neutral'
    
    def get_emotion_state(self):
        """根据情绪值返回对应的状态"""
        if self.emotion_value <= 3:
            return 'sad'
        elif self.emotion_value <= 6:
            return 'neutral'
        else:
            return 'happy'
    
    def update_emotion(self):
        """更新情绪值"""
        current_time = time.time()
        
        # 处理用户输入
        try:
            while not self.input_queue.empty():
                user_input = self.input_queue.get_nowait()
                if user_input == 'h' and current_time - self.last_input_time >= 1.0:
                    self.emotion_value = min(10.0, self.emotion_value + 1)
                    self.last_input_time = current_time
                    print(f"情绪值增加! 当前: {self.emotion_value:.1f}")
                elif user_input == 's' and current_time - self.last_input_time >= 1.0:
                    self.emotion_value = max(1.0, self.emotion_value - 1)
                    self.last_input_time = current_time
                    print(f"情绪值减少! 当前: {self.emotion_value:.1f}")
                elif user_input == 'q':
                    self.running = False
                    return
        except queue.Empty:
            pass
        
        # 自动回到平稳状态 (缓慢回到5)
        if current_time - self.last_input_time > 2.0:  # 3秒后开始自动调节
            if self.emotion_value > 5:
                self.emotion_value = max(5.0, self.emotion_value - 0.1)
            elif self.emotion_value < 5:
                self.emotion_value = min(5.0, self.emotion_value + 0.1)
        
        # 更新当前符号
        self.current_symbol = self.get_emotion_state()
    
    def add_user_input(self, user_input):
        """添加用户输入到队列"""
        self.input_queue.put(user_input)
    
    def decide(self):
        """基于当前状态做出决策"""
        # 根据情绪值调整行为
        if self.emotion_value >= 7:  # 高兴时
            self.move_randomly()
        elif self.emotion_value <= 3:  # 伤心时
            self.move_toward_center()
        else:  # 平稳时
            if random.random() > 0.5:
                self.move_randomly()
            else:
                self.move_toward_center()
    
    def move_toward_center(self):
        """向中心移动"""
        center = [self.width//2, self.height//2]
        for i in range(2):
            if self.position[i] < center[i]:
                self.position[i] += 1
            elif self.position[i] > center[i]:
                self.position[i] -= 1
    
    def move_randomly(self):
        """随机移动"""
        for i in range(2):
            self.position[i] += random.choice([-1, 0, 1])
            self.position[i] = max(0, min(self.width-5, self.position[i]))
    
    def render(self):
        """渲染当前状态到终端"""
        symbol = self.symbols[self.current_symbol]
        full_grid = np.zeros((self.height, self.width), dtype=int)
        
        # 将符号放置在当前位置
        y, x = self.position[1], self.position[0]
        for i in range(len(symbol)):
            for j in range(len(symbol[0])):
                if 0 <= y+i < self.height and 0 <= x+j < self.width:
                    full_grid[y+i][x+j] = symbol[i][j]
        
        # 打印到终端
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 30)
        print("   可交互ASCII AI")
        print("=" * 30)
        for row in full_grid:
            print(' '.join(['■' if cell else '·' for cell in row]))
        print("=" * 30)
        print(f"情绪状态: {self.current_symbol} ({self.emotion_value:.1f}/10)")
        
        # 显示情绪条
        emotion_bar = "情绪条: ["
        for i in range(10):
            if i < int(self.emotion_value):
                emotion_bar += "█"
            else:
                emotion_bar += "·"
        emotion_bar += "]"
        print(emotion_bar)
        
        print("-" * 30)
        print("操作: h=开心(+1) | s=难过(-1) | q=退出")
        print("自动回归: 情绪值会慢慢回到5")
    
    def update(self):
        """更新代理状态"""
        self.update_emotion()
        if self.running:
            self.decide()
            self.render()

def input_thread(agent):
    """处理用户输入的线程函数"""
    while agent.running:
        try:
            user_input = input(">>> ").strip().lower()
            if user_input in ['h', 's', 'q']:
                agent.add_user_input(user_input)
            elif user_input:
                print("无效输入! 请输入 h, s, 或 q")
        except (EOFError, KeyboardInterrupt):
            agent.add_user_input('q')
            break
        except Exception as e:
            print(f"输入错误: {e}")

# 使用示例
if __name__ == "__main__":
    print("🤖 ASCII AI 启动中...")
    print("这是一个可交互的ASCII人工智能")
    print("它有情绪系统，会根据你的操作改变情绪状态")
    print()
    
    agent = DotMatrixAgent()
    
    # 启动输入处理线程
    input_handler = threading.Thread(target=input_thread, args=(agent,), daemon=True)
    input_handler.start()
    
    try:
        # 主循环
        while agent.running:
            agent.update()
            time.sleep(0.5)  # 更新频率
    except KeyboardInterrupt:
        print("\n\n程序被用户中断...")
    except Exception as e:
        print(f"\n\n程序异常: {e}")
    finally:
        agent.running = False
        print("感谢使用 ASCII AI! 再见! 👋")
