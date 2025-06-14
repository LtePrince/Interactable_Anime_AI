"""
训练脚本 - 用于训练可交互动画AI模型
这是一个演示如何训练ConvLSTM U-Net模型的示例
"""

import numpy as np
import time
from typing import List, Dict, Tuple
import pickle
import os
from data_processor import DataProcessor, DataGenerator

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, frame_size=(5, 5)):
        self.frame_size = frame_size
        self.data_processor = DataProcessor(frame_size)
        self.data_generator = DataGenerator(frame_size)
        
        # 训练参数
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100
        
        # 损失历史
        self.loss_history = []
        self.accuracy_history = []
        
    def generate_training_data(self, num_sequences: int = 1000, 
                             sequence_length: int = 10) -> List[Dict]:
        """生成训练数据"""
        print(f"🎯 生成训练数据...")
        print(f"   序列数量: {num_sequences}")
        print(f"   序列长度: {sequence_length}")
        
        all_samples = []
        
        for i in range(num_sequences):
            if i % 100 == 0:
                print(f"   进度: {i}/{num_sequences}")
            
            sequence = self.data_generator.generate_synthetic_sequence(sequence_length)
            all_samples.extend(sequence)
        
        print(f"✅ 生成完成，总样本数: {len(all_samples)}")
        return all_samples
    
    def split_data(self, samples: List[Dict], 
                   train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """分割训练和验证数据"""
        np.random.shuffle(samples)
        split_idx = int(len(samples) * train_ratio)
        
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        print(f"📊 数据分割:")
        print(f"   训练集: {len(train_samples)} 样本")
        print(f"   验证集: {len(val_samples)} 样本")
        
        return train_samples, val_samples
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算损失（二元交叉熵）"""
        # 避免log(0)
        epsilon = 1e-7
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # 二元交叉熵损失
        loss = -np.mean(
            targets * np.log(predictions) + 
            (1 - targets) * np.log(1 - predictions)
        )
        
        return loss
    
    def compute_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算准确率"""
        pred_binary = (predictions > 0.5).astype(np.float32)
        accuracy = np.mean(pred_binary == targets)
        return accuracy
    
    def simple_forward_pass(self, batch_data: Dict) -> np.ndarray:
        """简化的前向传播（用于演示）"""
        frames = batch_data['frames']
        mouse = batch_data['mouse']
        emotion = batch_data['emotion']
        attention = batch_data['attention']
        
        batch_size = frames.shape[0]
        predictions = np.zeros((batch_size, *self.frame_size, 1), dtype=np.float32)
        
        for i in range(batch_size):
            # 简化的特征融合
            frame = frames[i, :, :, 0]
            mouse_features = mouse[i]
            emotion_features = emotion[i]
            attention_map = attention[i]
            
            # 提取关键特征
            mouse_x = mouse_features[0] * (self.frame_size[1] - 1)
            mouse_y = mouse_features[1] * (self.frame_size[0] - 1)
            mouse_click = mouse_features[2]
            
            happiness = emotion_features[0]
            arousal = emotion_features[1]
            engagement = emotion_features[2]
            
            # 初始化预测为当前帧的延续
            pred = frame * 0.6  # 帧延续性
            
            # 添加鼠标影响
            if mouse_click > 0.5:
                mouse_x_int = int(np.clip(mouse_x, 0, 4))
                mouse_y_int = int(np.clip(mouse_y, 0, 4))
                
                # 在鼠标位置及周围添加激活
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = mouse_y_int + dy, mouse_x_int + dx
                        if 0 <= ny < 5 and 0 <= nx < 5:
                            distance = abs(dy) + abs(dx)
                            influence = 0.8 * (1.0 - distance * 0.3)
                            pred[ny, nx] = min(1.0, pred[ny, nx] + influence)
            
            # 添加情绪影响
            emotion_base = happiness * 0.2  # 减少情绪基础影响
            pred += emotion_base
            
            # 添加激活度随机性
            if arousal > 0.6:
                noise_mask = np.random.random(self.frame_size) < (arousal * 0.1)
                pred[noise_mask] += 0.3
            
            # 应用注意力图
            pred = pred * (0.5 + 0.5 * attention_map)
            
            # 归一化和阈值处理
            pred = np.clip(pred, 0, 1)
            
            # 动态阈值
            threshold = 0.4 - engagement * 0.1
            threshold = max(0.2, min(0.6, threshold))
            
            pred = (pred > threshold).astype(np.float32)
            
            # 确保不是全黑
            if np.sum(pred) == 0:
                # 随机激活一个点
                rand_y, rand_x = np.random.randint(0, 5, 2)
                pred[rand_y, rand_x] = 1.0
            
            predictions[i, :, :, 0] = pred
        
        return predictions
    
    def train_epoch(self, train_samples: List[Dict]) -> Tuple[float, float]:
        """训练一个epoch"""
        np.random.shuffle(train_samples)
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for i in range(0, len(train_samples), self.batch_size):
            batch_samples = train_samples[i:i + self.batch_size]
            
            # 准备批次数据
            batch_data = self.data_processor.batch_process(batch_samples)
            
            # 前向传播
            predictions = self.simple_forward_pass(batch_data)
            targets = batch_data['targets']
            
            # 计算损失和准确率
            loss = self.compute_loss(predictions, targets)
            accuracy = self.compute_accuracy(predictions, targets)
            
            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def validate(self, val_samples: List[Dict]) -> Tuple[float, float]:
        """验证模型"""
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for i in range(0, len(val_samples), self.batch_size):
            batch_samples = val_samples[i:i + self.batch_size]
            
            # 准备批次数据
            batch_data = self.data_processor.batch_process(batch_samples)
            
            # 前向传播
            predictions = self.simple_forward_pass(batch_data)
            targets = batch_data['targets']
            
            # 计算损失和准确率
            loss = self.compute_loss(predictions, targets)
            accuracy = self.compute_accuracy(predictions, targets)
            
            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def train(self, num_sequences: int = 1000, save_path: str = "model_checkpoint.pkl"):
        """训练模型"""
        print("🚀 开始训练ConvLSTM可交互动画AI模型")
        print("=" * 60)
        
        # 生成训练数据
        samples = self.generate_training_data(num_sequences)
        train_samples, val_samples = self.split_data(samples)
        
        print(f"\n📈 训练参数:")
        print(f"   学习率: {self.learning_rate}")
        print(f"   批次大小: {self.batch_size}")
        print(f"   训练轮数: {self.num_epochs}")
        print(f"   帧尺寸: {self.frame_size}")
        
        print(f"\n🎯 开始训练...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_samples)
            
            # 验证
            val_loss, val_acc = self.validate(val_samples)
            
            # 记录历史
            self.loss_history.append((train_loss, val_loss))
            self.accuracy_history.append((train_acc, val_acc))
            
            epoch_time = time.time() - start_time
            
            # 打印进度
            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                print(f"Epoch {epoch+1:3d}/{self.num_epochs}: "
                      f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                      f"time={epoch_time:.2f}s")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(save_path)
        
        print(f"\n✅ 训练完成！")
        print(f"   最佳验证损失: {best_val_loss:.4f}")
        print(f"   模型已保存到: {save_path}")
        
        return self.loss_history, self.accuracy_history
    
    def save_model(self, filepath: str):
        """保存模型（简化版本，只保存训练历史）"""
        model_data = {
            'loss_history': self.loss_history,
            'accuracy_history': self.accuracy_history,
            'frame_size': self.frame_size,
            'training_params': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.loss_history = model_data['loss_history']
        self.accuracy_history = model_data['accuracy_history']
        self.frame_size = model_data['frame_size']
        
        if 'training_params' in model_data:
            params = model_data['training_params']
            self.learning_rate = params['learning_rate']
            self.batch_size = params['batch_size']
            self.num_epochs = params['num_epochs']
    
    def plot_training_history(self):
        """绘制训练历史（文本版本）"""
        if not self.loss_history:
            print("没有训练历史数据")
            return
        
        print("\n📊 训练历史:")
        print("-" * 80)
        print("Epoch  Train Loss  Val Loss    Train Acc   Val Acc")
        print("-" * 80)
        
        step = max(1, len(self.loss_history) // 20)  # 最多显示20行
        
        for i in range(0, len(self.loss_history), step):
            train_loss, val_loss = self.loss_history[i]
            train_acc, val_acc = self.accuracy_history[i]
            
            print(f"{i+1:5d}  {train_loss:10.4f}  {val_loss:9.4f}  "
                  f"{train_acc:10.4f}  {val_acc:8.4f}")
        
        print("-" * 80)
        
        # 显示最终结果
        final_train_loss, final_val_loss = self.loss_history[-1]
        final_train_acc, final_val_acc = self.accuracy_history[-1]
        
        print(f"最终结果:")
        print(f"  训练损失: {final_train_loss:.4f}")
        print(f"  验证损失: {final_val_loss:.4f}")
        print(f"  训练准确率: {final_train_acc:.4f}")
        print(f"  验证准确率: {final_val_acc:.4f}")

def run_training_demo():
    """运行训练演示"""
    print("🎓 ConvLSTM可交互动画AI 训练演示")
    print("=" * 60)
    
    # 创建训练器
    trainer = ModelTrainer(frame_size=(5, 5))
    
    # 运行训练（使用较小的数据集进行演示）
    print("📝 注意：这是一个简化的训练演示")
    print("   在实际应用中，您需要：")
    print("   1. 使用TensorFlow/PyTorch实现完整的ConvLSTM U-Net")
    print("   2. 收集真实的用户交互数据")
    print("   3. 使用GPU加速训练")
    print("   4. 实现更复杂的数据增强")
    print()
    
    # 开始训练
    loss_history, accuracy_history = trainer.train(
        num_sequences=100,  # 小数据集用于演示
        save_path="demo_model.pkl"
    )
    
    # 显示训练结果
    trainer.plot_training_history()
    
    print(f"\n💾 模型检查点已保存")
    print(f"📁 数据处理配置可保存为JSON文件")

def analyze_data_patterns():
    """分析数据模式"""
    print("\n🔍 数据模式分析:")
    print("-" * 40)
    
    # 创建数据生成器
    generator = DataGenerator(frame_size=(5, 5))
    processor = DataProcessor(frame_size=(5, 5))
    
    # 生成一些样本数据
    samples = generator.generate_synthetic_sequence(20)
    
    # 分析鼠标事件分布
    mouse_clicks = sum(1 for s in samples if s['mouse_event']['click'])
    print(f"鼠标点击频率: {mouse_clicks}/{len(samples)} ({mouse_clicks/len(samples)*100:.1f}%)")
    
    # 分析情绪变化
    happiness_values = [s['emotion_state']['happiness'] for s in samples]
    avg_happiness = np.mean(happiness_values)
    print(f"平均快乐度: {avg_happiness:.3f}")
    
    # 分析帧变化
    frame_changes = []
    for i in range(1, len(samples)):
        prev_frame = samples[i-1]['current_frame']
        curr_frame = samples[i]['current_frame']
        change = np.sum(np.abs(curr_frame - prev_frame))
        frame_changes.append(change)
    
    avg_change = np.mean(frame_changes)
    print(f"平均帧变化: {avg_change:.3f}")
    
    # 批处理演示
    batch_data = processor.batch_process(samples[:8])
    print(f"\n批处理数据维度:")
    for key, value in batch_data.items():
        print(f"  {key}: {value.shape}")

if __name__ == "__main__":
    # 运行训练演示
    run_training_demo()
    
    # 分析数据模式
    analyze_data_patterns()
    
    print(f"\n🎉 演示完成！")
    print(f"💡 下一步可以：")
    print(f"   1. 运行 SimpleConvLSTM_AI.py 测试交互")
    print(f"   2. 修改数据生成规则适应您的需求")
    print(f"   3. 实现完整的深度学习模型")
    print(f"   4. 收集真实用户数据进行训练")
