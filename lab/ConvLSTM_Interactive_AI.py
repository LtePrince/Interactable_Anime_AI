"""
基于ConvLSTM U-Net的可交互动画AI模型
输入：鼠标事件、情绪状态、当前动画帧
输出：5x5点阵动画帧
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.layers import ConvLSTM2D, Reshape, BatchNormalization, Activation
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
import time
import threading
import queue
from collections import deque

class MouseEvent:
    """鼠标事件类"""
    def __init__(self, x=0, y=0, click=False, event_type="none"):
        self.x = x  # 鼠标X坐标 (0-4)
        self.y = y  # 鼠标Y坐标 (0-4)
        self.click = click  # 是否点击
        self.event_type = event_type  # 事件类型: "move", "click", "none"
        
    def to_vector(self):
        """转换为特征向量"""
        return np.array([
            self.x / 4.0,  # 归一化坐标
            self.y / 4.0,
            1.0 if self.click else 0.0,
            1.0 if self.event_type == "move" else 0.0,
            1.0 if self.event_type == "click" else 0.0
        ], dtype=np.float32)

class EmotionState:
    """情绪状态类"""
    def __init__(self, happiness=0.5, arousal=0.5, engagement=0.5):
        self.happiness = happiness  # 快乐程度 (0-1)
        self.arousal = arousal      # 激活程度 (0-1)  
        self.engagement = engagement # 参与程度 (0-1)
        
    def to_vector(self):
        """转换为特征向量"""
        return np.array([
            self.happiness,
            self.arousal,
            self.engagement
        ], dtype=np.float32)

class InteractiveConvLSTM_UNet:
    """可交互的ConvLSTM U-Net模型"""
    
    def __init__(self, output_size=(5, 5)):
        self.output_size = output_size
        self.model = None
        self.build_model()
        
    def build_model(self):
        """构建模型"""
        # 输入定义
        # 1. 当前动画帧 (5x5x1)
        frame_input = Input(shape=(5, 5, 1), name='current_frame')
        
        # 2. 鼠标事件 (5维向量)
        mouse_input = Input(shape=(5,), name='mouse_event')
        
        # 3. 情绪状态 (3维向量)
        emotion_input = Input(shape=(3,), name='emotion_state')
        
        # 处理鼠标和情绪输入
        mouse_dense = Dense(16, activation='relu')(mouse_input)
        emotion_dense = Dense(16, activation='relu')(emotion_input)
        
        # 合并鼠标和情绪特征
        combined_features = concatenate([mouse_dense, emotion_dense])
        combined_features = Dense(32, activation='relu')(combined_features)
        combined_features = Dense(25, activation='relu')(combined_features)  # 5x5
        
        # 将特征重塑为5x5的特征图
        feature_map = Reshape((5, 5, 1))(combined_features)
        
        # 合并所有输入
        merged_input = concatenate([frame_input, feature_map], axis=-1)  # 5x5x2
        
        # 上采样到更大尺寸进行处理 (16x16)
        upsampled = UpSampling2D(size=(3, 3))(merged_input)  # 15x15
        upsampled = Conv2D(32, 3, padding='same', activation='relu')(upsampled)
        upsampled = Conv2D(32, 3, padding='same', activation='relu')(upsampled)
        upsampled = UpSampling2D(size=(2, 2))(upsampled)  # 30x30
        upsampled = Conv2D(64, 3, padding='same', activation='relu')(upsampled)
        
        # ConvLSTM层处理时序信息
        # 为ConvLSTM添加时间维度
        lstm_input = Reshape((1, 30, 30, 64))(upsampled)
        
        # ConvLSTM层
        conv_lstm1 = ConvLSTM2D(
            filters=32, 
            kernel_size=(3, 3), 
            padding='same', 
            return_sequences=True,
            activation='relu'
        )(lstm_input)
        
        conv_lstm2 = ConvLSTM2D(
            filters=16, 
            kernel_size=(3, 3), 
            padding='same', 
            return_sequences=False,
            activation='relu'
        )(conv_lstm1)
        
        # 下采样回到5x5
        downsampled = MaxPooling2D(pool_size=(2, 2))(conv_lstm2)  # 15x15
        downsampled = Conv2D(8, 3, padding='same', activation='relu')(downsampled)
        downsampled = MaxPooling2D(pool_size=(3, 3))(downsampled)  # 5x5
        
        # 输出层
        output = Conv2D(1, 1, activation='sigmoid', name='output_frame')(downsampled)
        
        # 创建模型
        self.model = Model(
            inputs=[frame_input, mouse_input, emotion_input],
            outputs=output
        )
        
        # 编译模型
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
    def predict(self, current_frame, mouse_event, emotion_state):
        """预测下一帧"""
        # 准备输入数据
        frame_input = np.expand_dims(current_frame, axis=0)  # (1, 5, 5, 1)
        mouse_input = np.expand_dims(mouse_event.to_vector(), axis=0)  # (1, 5)
        emotion_input = np.expand_dims(emotion_state.to_vector(), axis=0)  # (1, 3)
        
        # 预测
        prediction = self.model.predict([frame_input, mouse_input, emotion_input])
        
        # 后处理：二值化
        next_frame = (prediction[0, :, :, 0] > 0.5).astype(np.float32)
        return next_frame
    
    def get_model_summary(self):
        """获取模型摘要"""
        return self.model.summary()

class InteractiveAnimationAI:
    """可交互动画AI系统"""
    
    def __init__(self):
        self.model = InteractiveConvLSTM_UNet()
        self.current_frame = np.zeros((5, 5), dtype=np.float32)
        self.emotion_state = EmotionState()
        self.mouse_event = MouseEvent()
        self.running = True
        self.input_queue = queue.Queue()
        
        # 预定义一些基础动画帧
        self.base_frames = {
            'happy': np.array([
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1],
                [0, 1, 1, 1, 0]
            ], dtype=np.float32),
            'neutral': np.array([
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1]
            ], dtype=np.float32),
            'sad': np.array([
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1]
            ], dtype=np.float32)
        }
        
        # 初始化为中性表情
        self.current_frame = self.base_frames['neutral'].copy()
        
    def process_input(self, user_input):
        """处理用户输入"""
        if user_input.startswith('mouse:'):
            # 解析鼠标输入 "mouse:x,y,click"
            parts = user_input.split(':')[1].split(',')
            if len(parts) >= 2:
                try:
                    x = max(0, min(4, int(parts[0])))
                    y = max(0, min(4, int(parts[1])))
                    click = len(parts) > 2 and parts[2].lower() == 'true'
                    self.mouse_event = MouseEvent(x, y, click, "click" if click else "move")
                    
                    # 鼠标点击影响情绪
                    if click:
                        self.emotion_state.engagement = min(1.0, self.emotion_state.engagement + 0.1)
                        print(f"鼠标点击在 ({x}, {y})")
                except ValueError:
                    print("无效的鼠标输入格式")
                    
        elif user_input == 'h':
            # 增加快乐程度
            self.emotion_state.happiness = min(1.0, self.emotion_state.happiness + 0.2)
            self.emotion_state.arousal = min(1.0, self.emotion_state.arousal + 0.1)
            print(f"情绪变化：快乐度 {self.emotion_state.happiness:.2f}")
            
        elif user_input == 's':
            # 减少快乐程度
            self.emotion_state.happiness = max(0.0, self.emotion_state.happiness - 0.2)
            self.emotion_state.arousal = max(0.0, self.emotion_state.arousal - 0.1)
            print(f"情绪变化：快乐度 {self.emotion_state.happiness:.2f}")
            
        elif user_input == 'r':
            # 重置
            self.emotion_state = EmotionState()
            self.current_frame = self.base_frames['neutral'].copy()
            print("重置状态")
            
        elif user_input == 'q':
            self.running = False
            
    def update_frame(self):
        """更新动画帧"""
        # 使用模型预测下一帧（在实际应用中）
        # 这里我们使用规则来模拟模型输出
        
        # 根据情绪状态选择基础帧
        if self.emotion_state.happiness > 0.7:
            base = self.base_frames['happy']
        elif self.emotion_state.happiness < 0.3:
            base = self.base_frames['sad']
        else:
            base = self.base_frames['neutral']
            
        # 应用鼠标交互效果
        new_frame = base.copy()
        if self.mouse_event.click and self.mouse_event.event_type == "click":
            # 在鼠标位置添加效果
            x, y = self.mouse_event.x, self.mouse_event.y
            if 0 <= x < 5 and 0 <= y < 5:
                new_frame[y, x] = 1.0  # 点亮鼠标位置
                
                # 添加周围的光晕效果
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 5 and 0 <= ny < 5 and (dx != 0 or dy != 0):
                            new_frame[ny, nx] = max(new_frame[ny, nx], 0.3)
        
        # 添加激活程度的随机变化
        if self.emotion_state.arousal > 0.5:
            # 高激活时添加随机闪烁
            for i in range(5):
                for j in range(5):
                    if np.random.random() < 0.1 * self.emotion_state.arousal:
                        new_frame[i, j] = 1.0 - new_frame[i, j]
        
        self.current_frame = new_frame
        
        # 重置鼠标事件
        self.mouse_event = MouseEvent()
        
    def render(self):
        """渲染当前帧"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 40)
        print("   🤖 ConvLSTM 可交互动画AI")
        print("=" * 40)
        
        # 渲染5x5网格
        for row in self.current_frame:
            line = ' '.join(['██' if cell > 0.5 else '··' for cell in row])
            print(f"  {line}")
        
        print("=" * 40)
        print(f"情绪状态:")
        print(f"  快乐度: {self.emotion_state.happiness:.2f} {'😊' if self.emotion_state.happiness > 0.7 else '😐' if self.emotion_state.happiness > 0.3 else '😢'}")
        print(f"  激活度: {self.emotion_state.arousal:.2f}")
        print(f"  参与度: {self.emotion_state.engagement:.2f}")
        
        if self.mouse_event.event_type != "none":
            print(f"鼠标: ({self.mouse_event.x}, {self.mouse_event.y}) {'[点击]' if self.mouse_event.click else '[移动]'}")
        
        print("-" * 40)
        print("操作指南:")
        print("  h = 增加快乐度    s = 减少快乐度")
        print("  mouse:x,y,click = 鼠标操作 (例: mouse:2,3,true)")
        print("  r = 重置状态      q = 退出")
        print("-" * 40)
        
    def run(self):
        """运行AI系统"""
        print("🚀 ConvLSTM 可交互动画AI 启动中...")
        print("模型架构:")
        print("  输入: 当前帧(5x5) + 鼠标事件(5维) + 情绪状态(3维)")
        print("  输出: 下一帧(5x5)")
        print()
        
        # 显示模型摘要
        print("模型结构:")
        self.model.get_model_summary()
        print()
        
        # 启动输入处理线程
        input_thread = threading.Thread(target=self._input_handler, daemon=True)
        input_thread.start()
        
        try:
            while self.running:
                # 处理输入队列
                while not self.input_queue.empty():
                    try:
                        user_input = self.input_queue.get_nowait()
                        self.process_input(user_input)
                    except queue.Empty:
                        break
                
                # 更新帧
                self.update_frame()
                
                # 渲染
                self.render()
                
                # 情绪自然衰减
                self._decay_emotion()
                
                time.sleep(0.5)  # 控制更新频率
                
        except KeyboardInterrupt:
            print("\n程序被用户中断...")
        finally:
            self.running = False
            print("感谢使用 ConvLSTM 可交互动画AI! 👋")
    
    def _input_handler(self):
        """输入处理线程"""
        while self.running:
            try:
                user_input = input(">>> ").strip().lower()
                if user_input:
                    self.input_queue.put(user_input)
            except (EOFError, KeyboardInterrupt):
                self.input_queue.put('q')
                break
            except Exception as e:
                print(f"输入错误: {e}")
    
    def _decay_emotion(self):
        """情绪自然衰减"""
        decay_rate = 0.02
        
        # 快乐度向中性衰减
        if self.emotion_state.happiness > 0.5:
            self.emotion_state.happiness = max(0.5, self.emotion_state.happiness - decay_rate)
        elif self.emotion_state.happiness < 0.5:
            self.emotion_state.happiness = min(0.5, self.emotion_state.happiness + decay_rate)
        
        # 激活度和参与度衰减
        self.emotion_state.arousal = max(0.0, self.emotion_state.arousal - decay_rate)
        self.emotion_state.engagement = max(0.0, self.emotion_state.engagement - decay_rate)

if __name__ == "__main__":
    # 创建并运行可交互动画AI
    ai = InteractiveAnimationAI()
    ai.run()
