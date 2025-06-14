"""
简化版可交互动画AI模型
基于ConvLSTM U-Net的思想，使用纯NumPy实现
输入：鼠标事件、情绪状态、当前动画帧
输出：5x5点阵动画帧
"""

import numpy as np
import time
import threading
import queue
from collections import deque
import random
import pickle
import os

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

class SimpleConvLSTMCell:
    """简化的ConvLSTM单元（用于演示）"""
    def __init__(self, input_size=(5, 5), hidden_size=8):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 简化的权重矩阵（更好的初始化）
        np.random.seed(42)  # 固定随机种子
        
        # 使用Xavier初始化
        scale = np.sqrt(2.0 / (hidden_size + hidden_size))
        self.W_f = np.random.randn(hidden_size, hidden_size) * scale  # 遗忘门权重
        self.W_i = np.random.randn(hidden_size, hidden_size) * scale  # 输入门权重
        self.W_o = np.random.randn(hidden_size, hidden_size) * scale  # 输出门权重
        self.W_c = np.random.randn(hidden_size, hidden_size) * scale  # 候选值权重
        
        # 偏置（遗忘门偏置设为正值以避免梯度消失）
        self.b_f = np.ones(hidden_size) * 1.0  # 遗忘门偏置为正
        self.b_i = np.zeros(hidden_size)
        self.b_o = np.zeros(hidden_size)
        self.b_c = np.zeros(hidden_size)
        
        # 隐藏状态和细胞状态
        self.h = np.zeros((input_size[0], input_size[1], hidden_size))
        self.c = np.zeros((input_size[0], input_size[1], hidden_size))
        
    def sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        """Tanh激活函数"""
        return np.tanh(np.clip(x, -500, 500))
    
    def forward(self, input_data):
        """前向传播（简化版本）"""
        # 将输入数据转换为合适的形状
        if input_data.shape != self.input_size:
            input_data = np.resize(input_data, self.input_size)
            
        # 简化的ConvLSTM计算
        # 这里我们使用简单的线性组合和激活函数
        
        # 计算门控值（简化）
        combined = np.mean(input_data) + np.mean(self.h)
        
        f_gate = self.sigmoid(combined + self.b_f[0])  # 遗忘门
        i_gate = self.sigmoid(combined + self.b_i[0])  # 输入门
        o_gate = self.sigmoid(combined + self.b_o[0])  # 输出门
        c_candidate = self.tanh(combined + self.b_c[0])  # 候选值
        
        # 更新细胞状态
        self.c = f_gate * self.c + i_gate * c_candidate
        
        # 更新隐藏状态
        self.h = o_gate * self.tanh(self.c)
        
        return self.h

class SimpleInteractiveModel:
    """简化的可交互模型"""
    
    def __init__(self, output_size=(5, 5)):
        self.output_size = output_size
        self.conv_lstm = SimpleConvLSTMCell(output_size)
        
        # 特征融合权重（确保总和为1）
        self.mouse_weight = 0.4
        self.emotion_weight = 0.2
        self.frame_weight = 0.4
        
    def predict(self, current_frame, mouse_event, emotion_state):
        """预测下一帧"""
        # 获取输入特征
        mouse_vec = mouse_event.to_vector()
        emotion_vec = emotion_state.to_vector()
        
        # 特征融合
        # 1. 鼠标影响：在鼠标位置附近增强
        mouse_influence = np.zeros(self.output_size)
        if mouse_event.event_type != "none":
            x, y = mouse_event.x, mouse_event.y
            # 确保坐标在有效范围内
            x = max(0, min(4, x))
            y = max(0, min(4, y))
            
            for i in range(max(0, y-1), min(5, y+2)):
                for j in range(max(0, x-1), min(5, x+2)):
                    distance = abs(i - y) + abs(j - x)
                    influence = 0.8 if mouse_event.click else 0.3
                    mouse_influence[i, j] = influence * max(0.1, 1.0 - distance * 0.5)
        
        # 2. 情绪影响：全局模式（修复范围）
        emotion_base = emotion_state.happiness * 0.3  # 降低基础影响
        emotion_influence = np.ones(self.output_size) * emotion_base
        
        # 添加激活度的随机性（减少噪声）
        if emotion_state.arousal > 0.6:
            noise = np.random.random(self.output_size) * emotion_state.arousal * 0.1
            emotion_influence += noise
        
        # 3. 当前帧的延续性（增加延续性）
        frame_influence = current_frame * 0.7  # 帧间连续性
        
        # 综合所有影响
        combined_input = (
            frame_influence * self.frame_weight +
            mouse_influence * self.mouse_weight +
            emotion_influence * self.emotion_weight
        )
        
        # 通过ConvLSTM处理
        lstm_output = self.conv_lstm.forward(combined_input)
        
        # 提取输出特征并转换为二值化结果
        output_frame = np.mean(lstm_output, axis=2)  # 平均所有通道
        
        # 归一化到[0,1]范围
        if output_frame.max() > output_frame.min():
            output_frame = (output_frame - output_frame.min()) / (output_frame.max() - output_frame.min())
        
        # 应用动态阈值
        base_threshold = 0.4  # 降低基础阈值
        engagement_adjust = emotion_state.engagement * 0.1  # 减少参与度影响
        threshold = base_threshold - engagement_adjust
        
        # 确保阈值在合理范围内
        threshold = max(0.2, min(0.7, threshold))
        
        output_frame = (output_frame > threshold).astype(np.float32)
        
        return output_frame

class InteractiveAnimationAI:
    """可交互动画AI系统"""
    
    def __init__(self, load_model_path=None):
        self.model = SimpleInteractiveModel()
        self.current_frame = np.zeros((5, 5), dtype=np.float32)
        self.emotion_state = EmotionState()
        self.mouse_event = MouseEvent()
        self.running = True
        self.input_queue = queue.Queue()
        self.frame_history = deque(maxlen=10)  # 保存历史帧
        
        # 尝试加载模型（如果提供了路径）
        if load_model_path and os.path.exists(load_model_path):
            try:
                self.load_model(load_model_path)
                print(f"✅ 成功加载模型: {load_model_path}")
            except Exception as e:
                print(f"⚠️  模型加载失败: {e}")
                print("使用默认参数继续运行...")
        elif load_model_path:
            print(f"⚠️  模型文件未找到: {load_model_path}")
            print("使用默认参数继续运行...")
        
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
            ], dtype=np.float32),
            'excited': np.array([
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1]
            ], dtype=np.float32)
        }
        
        # 初始化为中性表情
        self.current_frame = self.base_frames['neutral'].copy()
        self.frame_history.append(self.current_frame.copy())
        
        # 尝试加载预训练模型
        self._load_model()
        
    def _load_model(self):
        """加载预训练模型（如果可用）"""
        try:
            with open("model_weights.pkl", "rb") as f:
                weights = pickle.load(f)
                self.model.conv_lstm.W_f = weights['W_f']
                self.model.conv_lstm.W_i = weights['W_i']
                self.model.conv_lstm.W_o = weights['W_o']
                self.model.conv_lstm.W_c = weights['W_c']
                self.model.conv_lstm.b_f = weights['b_f']
                self.model.conv_lstm.b_i = weights['b_i']
                self.model.conv_lstm.b_o = weights['b_o']
                self.model.conv_lstm.b_c = weights['b_c']
                print("✅ 成功加载预训练模型")
        except Exception as e:
            print(f"❌ 加载预训练模型时出错: {e}")
    
    def load_model(self, filepath: str):
        """加载模型参数"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # 如果有训练参数，可以在这里应用
        if 'training_params' in model_data:
            params = model_data['training_params']
            print(f"📊 加载的训练参数:")
            print(f"   学习率: {params.get('learning_rate', 'N/A')}")
            print(f"   批次大小: {params.get('batch_size', 'N/A')}")
            print(f"   训练轮数: {params.get('num_epochs', 'N/A')}")
        
        # 如果有损失历史，显示训练效果
        if 'loss_history' in model_data and model_data['loss_history']:
            final_train_loss, final_val_loss = model_data['loss_history'][-1]
            final_train_acc, final_val_acc = model_data['accuracy_history'][-1]
            print(f"📈 训练结果:")
            print(f"   最终训练损失: {final_train_loss:.4f}")
            print(f"   最终验证损失: {final_val_loss:.4f}")
            print(f"   最终训练准确率: {final_train_acc:.4f}")
            print(f"   最终验证准确率: {final_val_acc:.4f}")
            
            # 基于训练结果调整模型行为（示例）
            if final_val_acc > 0.9:
                print("🎯 模型训练效果良好，使用较高的响应灵敏度")
                self.model.mouse_weight = 0.5  # 增加鼠标响应
            else:
                print("⚠️  模型训练效果一般，使用保守的响应参数")
                self.model.mouse_weight = 0.3  # 保守的鼠标响应
    
    def _save_model(self):
        """保存当前模型状态"""
        try:
            model_data = {
                'model_params': {
                    'mouse_weight': self.model.mouse_weight,
                    'emotion_weight': self.model.emotion_weight,
                    'frame_weight': self.model.frame_weight,
                },
                'emotion_state': {
                    'happiness': self.emotion_state.happiness,
                    'arousal': self.emotion_state.arousal,
                    'engagement': self.emotion_state.engagement,
                },
                'frame_history': list(self.frame_history),
                'save_time': time.time()
            }
            
            with open('session_save.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            print("💾 会话状态已保存到 session_save.pkl")
        except Exception as e:
            print(f"保存失败: {e}")
    
    def load_session(self, filepath='session_save.pkl'):
        """加载会话状态"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    session_data = pickle.load(f)
                
                # 恢复模型参数
                if 'model_params' in session_data:
                    params = session_data['model_params']
                    self.model.mouse_weight = params.get('mouse_weight', 0.4)
                    self.model.emotion_weight = params.get('emotion_weight', 0.2)
                    self.model.frame_weight = params.get('frame_weight', 0.4)
                
                # 恢复情绪状态
                if 'emotion_state' in session_data:
                    emotion = session_data['emotion_state']
                    self.emotion_state.happiness = emotion.get('happiness', 0.5)
                    self.emotion_state.arousal = emotion.get('arousal', 0.5)
                    self.emotion_state.engagement = emotion.get('engagement', 0.5)
                
                # 恢复帧历史
                if 'frame_history' in session_data:
                    for frame in session_data['frame_history']:
                        self.frame_history.append(np.array(frame))
                    if self.frame_history:
                        self.current_frame = self.frame_history[-1].copy()
                
                print(f"✅ 会话状态已恢复")
                return True
            except Exception as e:
                print(f"会话加载失败: {e}")
                return False
        return False
    
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
                        self.emotion_state.engagement = min(1.0, self.emotion_state.engagement + 0.15)
                        self.emotion_state.arousal = min(1.0, self.emotion_state.arousal + 0.1)
                        print(f"🖱️ 鼠标点击在 ({x}, {y})")
                    else:
                        self.emotion_state.engagement = min(1.0, self.emotion_state.engagement + 0.05)
                        print(f"🖱️ 鼠标移动到 ({x}, {y})")
                except ValueError:
                    print("❌ 无效的鼠标输入格式")
                    
        elif user_input == 'h':
            # 增加快乐程度
            self.emotion_state.happiness = min(1.0, self.emotion_state.happiness + 0.2)
            self.emotion_state.arousal = min(1.0, self.emotion_state.arousal + 0.1)
            print(f"😊 情绪变化：快乐度 {self.emotion_state.happiness:.2f}")
            
        elif user_input == 's':
            # 减少快乐程度
            self.emotion_state.happiness = max(0.0, self.emotion_state.happiness - 0.2)
            self.emotion_state.arousal = max(0.0, self.emotion_state.arousal - 0.1)
            print(f"😢 情绪变化：快乐度 {self.emotion_state.happiness:.2f}")
            
        elif user_input == 'e':
            # 兴奋模式
            self.emotion_state.arousal = min(1.0, self.emotion_state.arousal + 0.3)
            self.emotion_state.engagement = min(1.0, self.emotion_state.engagement + 0.2)
            print(f"🎉 兴奋模式！激活度: {self.emotion_state.arousal:.2f}")
            
        elif user_input == 'c':
            # 冷静模式
            self.emotion_state.arousal = max(0.0, self.emotion_state.arousal - 0.3)
            print(f"😌 冷静模式！激活度: {self.emotion_state.arousal:.2f}")
            
        elif user_input == 'r':
            # 重置
            self.emotion_state = EmotionState()
            self.current_frame = self.base_frames['neutral'].copy()
            self.mouse_event = MouseEvent()
            print("🔄 重置状态")
            
        elif user_input == 'demo':
            # 演示模式
            self._run_demo()
            
        elif user_input == 'q':
            self.running = False
            
    def update_frame(self):
        """使用模型更新动画帧"""
        # 使用AI模型预测下一帧
        predicted_frame = self.model.predict(self.current_frame, self.mouse_event, self.emotion_state)
        
        # 融合模型预测和基础表情（减少基础表情影响）
        if self.emotion_state.happiness > 0.8:
            # 非常开心时，使用开心的基础模式
            base = self.base_frames['happy']
            next_frame = 0.8 * predicted_frame + 0.2 * base
            
        elif self.emotion_state.happiness < 0.2:
            # 非常难过时，使用难过的基础模式
            base = self.base_frames['sad']
            next_frame = 0.8 * predicted_frame + 0.2 * base
            
        elif self.emotion_state.arousal > 0.8:
            # 非常兴奋时，使用兴奋模式
            base = self.base_frames['excited']
            next_frame = 0.7 * predicted_frame + 0.3 * base
        else:
            # 其他情况主要使用模型预测
            next_frame = predicted_frame
        
        # 添加一些动态效果
        if self.mouse_event.event_type == "click":
            # 鼠标点击时增强该位置
            x, y = self.mouse_event.x, self.mouse_event.y
            if 0 <= x < 5 and 0 <= y < 5:
                next_frame[y, x] = 1.0
        
        # 确保帧不是全白或全黑（添加最小活动）
        if np.sum(next_frame) == 0:
            # 如果全黑，随机激活一些点
            random_points = np.random.choice(25, size=2, replace=False)
            for point in random_points:
                row, col = divmod(point, 5)
                next_frame[row, col] = 1.0
        elif np.sum(next_frame) >= 24:
            # 如果几乎全白，随机关闭一些点
            active_points = np.where(next_frame == 1.0)
            if len(active_points[0]) > 2:
                idx = np.random.choice(len(active_points[0]), size=2, replace=False)
                for i in idx:
                    next_frame[active_points[0][i], active_points[1][i]] = 0.0
        
        # 二值化
        next_frame = (next_frame > 0.5).astype(np.float32)
        
        # 更新当前帧
        self.current_frame = next_frame
        self.frame_history.append(self.current_frame.copy())
        
        # 重置鼠标事件（延迟重置以保持影响）
        if self.mouse_event.event_type == "click":
            # 点击事件保持一帧
            self.mouse_event.event_type = "move"
        else:
            self.mouse_event = MouseEvent()
        
    def render(self):
        """渲染当前帧"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 50)
        print("   🤖 ConvLSTM 可交互动画AI - 简化版")
        print("=" * 50)
        
        # 渲染5x5网格
        for row in self.current_frame:
            line = ' '.join(['██' if cell > 0.5 else '··' for cell in row])
            print(f"    {line}")
        
        print("=" * 50)
        
        # 情绪状态可视化
        print(f"📊 情绪状态:")
        happiness_bar = self._create_bar(self.emotion_state.happiness, "😊", "😐", "😢")
        arousal_bar = self._create_bar(self.emotion_state.arousal, "🔥", "🌿", "😴")
        engagement_bar = self._create_bar(self.emotion_state.engagement, "🎯", "👀", "💤")
        
        print(f"  快乐度: {happiness_bar} ({self.emotion_state.happiness:.2f})")
        print(f"  激活度: {arousal_bar} ({self.emotion_state.arousal:.2f})")
        print(f"  参与度: {engagement_bar} ({self.emotion_state.engagement:.2f})")
        
        # 显示鼠标状态
        if self.mouse_event.event_type != "none":
            print(f"🖱️  鼠标: ({self.mouse_event.x}, {self.mouse_event.y}) {'[点击]' if self.mouse_event.click else '[移动]'}")
        
        # 显示模型信息
        print(f"🧠 模型状态: ConvLSTM隐藏状态形状 {self.model.conv_lstm.h.shape}")
        print(f"📈 历史帧数: {len(self.frame_history)}")
        
        print("-" * 50)
        print("📝 操作指南:")
        print("  h/s = 快乐/难过    e/c = 兴奋/冷静")
        print("  mouse:x,y,click = 鼠标操作 (例: mouse:2,3,true)")
        print("  r = 重置    demo = 演示    q = 退出")
        print("-" * 50)
        
    def _create_bar(self, value, high_emoji, mid_emoji, low_emoji):
        """创建情绪条"""
        bar_length = 10
        filled = int(value * bar_length)
        bar = "█" * filled + "·" * (bar_length - filled)
        
        if value > 0.7:
            emoji = high_emoji
        elif value > 0.3:
            emoji = mid_emoji
        else:
            emoji = low_emoji
            
        return f"{emoji} [{bar}]"
        
    def _run_demo(self):
        """运行演示模式"""
        print("🎬 开始演示模式...")
        
        demo_actions = [
            ("mouse:2,2,true", "鼠标点击中心"),
            ("h", "增加快乐度"),
            ("e", "激活兴奋模式"),
            ("mouse:0,0,true", "点击左上角"),
            ("mouse:4,4,true", "点击右下角"),
            ("s", "减少快乐度"),
            ("c", "冷静模式"),
            ("r", "重置状态")
        ]
        
        for action, description in demo_actions:
            print(f"🎭 演示: {description}")
            self.process_input(action)
            self.update_frame()
            self.render()
            time.sleep(1.5)
            
        print("🎬 演示完成！")
        
    def run(self):
        """运行AI系统"""
        print("🚀 ConvLSTM 可交互动画AI 启动中...")
        print("🏗️  模型架构:")
        print("   输入: 当前帧(5x5) + 鼠标事件(5维) + 情绪状态(3维)")
        print("   处理: 简化ConvLSTM + 特征融合")
        print("   输出: 下一帧(5x5)")
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
                
                time.sleep(0.8)  # 控制更新频率
                
        except KeyboardInterrupt:
            print("\n程序被用户中断...")
        finally:
            self.running = False
            print("感谢使用 ConvLSTM 可交互动画AI! 👋")
            self._save_model()
    
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
        self.emotion_state.arousal = max(0.0, self.emotion_state.arousal - decay_rate * 1.5)
        self.emotion_state.engagement = max(0.0, self.emotion_state.engagement - decay_rate)
    
    def _save_model(self):
        """保存模型权重"""
        try:
            weights = {
                'W_f': self.model.conv_lstm.W_f,
                'W_i': self.model.conv_lstm.W_i,
                'W_o': self.model.conv_lstm.W_o,
                'W_c': self.model.conv_lstm.W_c,
                'b_f': self.model.conv_lstm.b_f,
                'b_i': self.model.conv_lstm.b_i,
                'b_o': self.model.conv_lstm.b_o,
                'b_c': self.model.conv_lstm.b_c
            }
            with open("model_weights.pkl", "wb") as f:
                pickle.dump(weights, f)
            print("💾 模型权重已保存")
        except Exception as e:
            print(f"❌ 保存模型权重时出错: {e}")

if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    load_model_path = None
    load_session = False
    
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith('--model='):
                load_model_path = arg.split('=')[1]
            elif arg == '--load-session':
                load_session = True
            elif arg == '--help':
                print("🤖 ConvLSTM 可交互动画AI")
                print("用法:")
                print("  python SimpleConvLSTM_AI.py                    # 默认运行")
                print("  python SimpleConvLSTM_AI.py --model=demo_model.pkl  # 加载训练模型")
                print("  python SimpleConvLSTM_AI.py --load-session          # 恢复上次会话")
                print("  python SimpleConvLSTM_AI.py --help                  # 显示帮助")
                sys.exit(0)
    
    # 创建并运行可交互动画AI
    ai = InteractiveAnimationAI(load_model_path=load_model_path)
    
    # 如果需要，加载会话
    if load_session:
        ai.load_session()
    
    print(f"🎮 运行模式:")
    if load_model_path:
        print(f"   训练模型: {load_model_path}")
    if load_session:
        print(f"   会话恢复: 启用")
    print(f"   退出时自动保存会话状态")
    print()
    
    ai.run()
