"""
数据预处理模块
用于处理可交互动画AI的输入数据：鼠标事件、情绪状态、动画帧
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
import json

class DataProcessor:
    """数据预处理器"""
    
    def __init__(self, frame_size=(5, 5)):
        self.frame_size = frame_size
        self.normalization_params = {
            'mouse_x_range': (0, frame_size[1] - 1),
            'mouse_y_range': (0, frame_size[0] - 1),
            'emotion_range': (0.0, 1.0)
        }
        
    def normalize_mouse_coordinates(self, x: int, y: int) -> Tuple[float, float]:
        """归一化鼠标坐标"""
        norm_x = x / (self.frame_size[1] - 1)
        norm_y = y / (self.frame_size[0] - 1)
        return norm_x, norm_y
    
    def denormalize_mouse_coordinates(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """反归一化鼠标坐标"""
        x = int(norm_x * (self.frame_size[1] - 1))
        y = int(norm_y * (self.frame_size[0] - 1))
        return x, y
    
    def encode_mouse_event(self, x: int, y: int, click: bool, event_type: str) -> np.ndarray:
        """编码鼠标事件为特征向量"""
        norm_x, norm_y = self.normalize_mouse_coordinates(x, y)
        
        # 事件类型编码
        type_encoding = {
            'none': [0, 0, 0],
            'move': [1, 0, 0],
            'click': [0, 1, 0],
            'drag': [0, 0, 1]
        }
        
        type_vec = type_encoding.get(event_type, [0, 0, 0])
        
        return np.array([
            norm_x,
            norm_y,
            1.0 if click else 0.0,
            *type_vec
        ], dtype=np.float32)
    
    def encode_emotion_state(self, happiness: float, arousal: float, engagement: float) -> np.ndarray:
        """编码情绪状态为特征向量"""
        # 确保值在[0,1]范围内
        happiness = np.clip(happiness, 0.0, 1.0)
        arousal = np.clip(arousal, 0.0, 1.0)
        engagement = np.clip(engagement, 0.0, 1.0)
        
        # 计算复合情绪特征
        valence = happiness  # 效价
        energy = arousal     # 能量
        attention = engagement  # 注意力
        
        # 添加派生特征
        excitement = (happiness + arousal) / 2  # 兴奋度
        calmness = (happiness + (1 - arousal)) / 2  # 平静度
        
        return np.array([
            valence,
            energy,
            attention,
            excitement,
            calmness
        ], dtype=np.float32)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理动画帧"""
        # 确保帧尺寸正确
        if frame.shape != self.frame_size:
            frame = np.resize(frame, self.frame_size)
        
        # 归一化到[0,1]
        frame = frame.astype(np.float32)
        if frame.max() > 1.0:
            frame = frame / frame.max()
        
        # 添加通道维度
        frame = np.expand_dims(frame, axis=-1)
        
        return frame
    
    def create_spatial_attention_map(self, mouse_x: int, mouse_y: int, 
                                   attention_radius: float = 1.5) -> np.ndarray:
        """创建基于鼠标位置的空间注意力图"""
        attention_map = np.zeros(self.frame_size, dtype=np.float32)
        
        for i in range(self.frame_size[0]):
            for j in range(self.frame_size[1]):
                distance = np.sqrt((i - mouse_y)**2 + (j - mouse_x)**2)
                attention = np.exp(-distance / attention_radius)
                attention_map[i, j] = attention
        
        # 归一化
        attention_map = attention_map / (attention_map.max() + 1e-8)
        
        return attention_map
    
    def create_temporal_features(self, frame_history: List[np.ndarray]) -> np.ndarray:
        """从历史帧创建时序特征"""
        if len(frame_history) == 0:
            return np.zeros((*self.frame_size, 1), dtype=np.float32)
        
        # 计算帧间差分
        if len(frame_history) >= 2:
            frame_diff = frame_history[-1] - frame_history[-2]
        else:
            frame_diff = np.zeros_like(frame_history[-1])
        
        # 计算运动强度
        motion_intensity = np.abs(frame_diff).mean()
        
        # 创建运动地图
        motion_map = np.abs(frame_diff)
        
        # 归一化
        if motion_map.max() > 0:
            motion_map = motion_map / motion_map.max()
        
        # 添加通道维度
        motion_map = np.expand_dims(motion_map, axis=-1)
        
        return motion_map
    
    def augment_frame(self, frame: np.ndarray, augmentation_type: str = 'none') -> np.ndarray:
        """数据增强"""
        if augmentation_type == 'noise':
            # 添加随机噪声
            noise = np.random.normal(0, 0.1, frame.shape)
            frame = np.clip(frame + noise, 0, 1)
        
        elif augmentation_type == 'flip_h':
            # 水平翻转
            frame = np.flip(frame, axis=1)
        
        elif augmentation_type == 'flip_v':
            # 垂直翻转
            frame = np.flip(frame, axis=0)
        
        elif augmentation_type == 'rotate':
            # 90度旋转
            frame = np.rot90(frame)
        
        return frame.astype(np.float32)
    
    def create_training_sample(self, 
                             current_frame: np.ndarray,
                             mouse_event: Dict,
                             emotion_state: Dict,
                             next_frame: np.ndarray,
                             frame_history: Optional[List[np.ndarray]] = None) -> Dict:
        """创建训练样本"""
        
        # 预处理输入
        processed_frame = self.preprocess_frame(current_frame)
        mouse_features = self.encode_mouse_event(
            mouse_event['x'], mouse_event['y'], 
            mouse_event['click'], mouse_event['type']
        )
        emotion_features = self.encode_emotion_state(
            emotion_state['happiness'], 
            emotion_state['arousal'], 
            emotion_state['engagement']
        )
        
        # 创建注意力图
        attention_map = self.create_spatial_attention_map(
            mouse_event['x'], mouse_event['y']
        )
        
        # 创建时序特征
        if frame_history:
            temporal_features = self.create_temporal_features(frame_history)
        else:
            temporal_features = np.zeros((*self.frame_size, 1), dtype=np.float32)
        
        # 预处理目标
        target_frame = self.preprocess_frame(next_frame)
        
        return {
            'current_frame': processed_frame,
            'mouse_features': mouse_features,
            'emotion_features': emotion_features,
            'attention_map': attention_map,
            'temporal_features': temporal_features,
            'target_frame': target_frame
        }
    
    def batch_process(self, data_list: List[Dict]) -> Dict[str, np.ndarray]:
        """批处理数据"""
        batch_size = len(data_list)
        
        # 初始化批次数据
        batch_frames = np.zeros((batch_size, *self.frame_size, 1), dtype=np.float32)
        batch_mouse = np.zeros((batch_size, 6), dtype=np.float32)  # 6维鼠标特征
        batch_emotion = np.zeros((batch_size, 5), dtype=np.float32)  # 5维情绪特征
        batch_attention = np.zeros((batch_size, *self.frame_size), dtype=np.float32)
        batch_temporal = np.zeros((batch_size, *self.frame_size, 1), dtype=np.float32)
        batch_targets = np.zeros((batch_size, *self.frame_size, 1), dtype=np.float32)
        
        # 填充批次数据
        for i, sample in enumerate(data_list):
            processed_sample = self.create_training_sample(**sample)
            
            batch_frames[i] = processed_sample['current_frame']
            batch_mouse[i] = processed_sample['mouse_features']
            batch_emotion[i] = processed_sample['emotion_features']
            batch_attention[i] = processed_sample['attention_map']
            batch_temporal[i] = processed_sample['temporal_features']
            batch_targets[i] = processed_sample['target_frame']
        
        return {
            'frames': batch_frames,
            'mouse': batch_mouse,
            'emotion': batch_emotion,
            'attention': batch_attention,
            'temporal': batch_temporal,
            'targets': batch_targets
        }
    
    def save_preprocessing_config(self, filepath: str):
        """保存预处理配置"""
        config = {
            'frame_size': self.frame_size,
            'normalization_params': self.normalization_params
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_preprocessing_config(self, filepath: str):
        """加载预处理配置"""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.frame_size = tuple(config['frame_size'])
        self.normalization_params = config['normalization_params']

class DataGenerator:
    """数据生成器（用于生成合成训练数据）"""
    
    def __init__(self, frame_size=(5, 5)):
        self.frame_size = frame_size
        self.processor = DataProcessor(frame_size)
        
    def generate_synthetic_sequence(self, sequence_length: int = 10) -> List[Dict]:
        """生成合成的交互序列"""
        sequence = []
        
        # 初始状态
        current_frame = np.random.choice([0, 1], size=self.frame_size).astype(np.float32)
        emotion_state = {
            'happiness': np.random.uniform(0.2, 0.8),
            'arousal': np.random.uniform(0.2, 0.8),
            'engagement': np.random.uniform(0.2, 0.8)
        }
        
        for step in range(sequence_length):
            # 生成随机鼠标事件
            mouse_event = {
                'x': np.random.randint(0, self.frame_size[1]),
                'y': np.random.randint(0, self.frame_size[0]),
                'click': np.random.choice([True, False], p=[0.3, 0.7]),
                'type': np.random.choice(['move', 'click', 'none'], p=[0.4, 0.3, 0.3])
            }
            
            # 生成下一帧（基于简单规则）
            next_frame = self._generate_next_frame(current_frame, mouse_event, emotion_state)
            
            # 创建样本
            sample = {
                'current_frame': current_frame.copy(),
                'mouse_event': mouse_event.copy(),
                'emotion_state': emotion_state.copy(),
                'next_frame': next_frame.copy()
            }
            
            sequence.append(sample)
            
            # 更新状态
            current_frame = next_frame
            emotion_state = self._update_emotion_state(emotion_state, mouse_event)
        
        return sequence
    
    def _generate_next_frame(self, current_frame: np.ndarray, 
                           mouse_event: Dict, emotion_state: Dict) -> np.ndarray:
        """基于规则生成下一帧"""
        next_frame = current_frame.copy()
        
        # 鼠标影响
        if mouse_event['click']:
            x, y = mouse_event['x'], mouse_event['y']
            next_frame[y, x] = 1.0
            
            # 周围扩散
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.frame_size[1] and 0 <= ny < self.frame_size[0]:
                        next_frame[ny, nx] = max(next_frame[ny, nx], 0.5)
        
        # 情绪影响
        if emotion_state['arousal'] > 0.7:
            # 高激活时添加随机噪声
            noise_mask = np.random.random(self.frame_size) < 0.1
            next_frame[noise_mask] = 1.0 - next_frame[noise_mask]
        
        # 衰减效应
        next_frame = next_frame * 0.9
        
        return (next_frame > 0.3).astype(np.float32)
    
    def _update_emotion_state(self, emotion_state: Dict, mouse_event: Dict) -> Dict:
        """更新情绪状态"""
        new_state = emotion_state.copy()
        
        # 鼠标交互影响情绪
        if mouse_event['click']:
            new_state['engagement'] = min(1.0, new_state['engagement'] + 0.1)
            new_state['arousal'] = min(1.0, new_state['arousal'] + 0.05)
        
        # 自然衰减
        new_state['arousal'] = max(0.0, new_state['arousal'] - 0.02)
        new_state['engagement'] = max(0.0, new_state['engagement'] - 0.01)
        
        return new_state

# 使用示例
if __name__ == "__main__":
    # 创建数据处理器
    processor = DataProcessor(frame_size=(5, 5))
    
    # 测试数据编码
    mouse_features = processor.encode_mouse_event(2, 3, True, 'click')
    print("鼠标特征:", mouse_features)
    
    emotion_features = processor.encode_emotion_state(0.8, 0.6, 0.7)
    print("情绪特征:", emotion_features)
    
    # 测试帧预处理
    test_frame = np.random.randint(0, 2, (5, 5))
    processed_frame = processor.preprocess_frame(test_frame)
    print("原始帧:\n", test_frame)
    print("处理后帧形状:", processed_frame.shape)
    
    # 创建注意力图
    attention_map = processor.create_spatial_attention_map(2, 3)
    print("注意力图:\n", attention_map)
    
    # 生成合成数据
    generator = DataGenerator()
    synthetic_sequence = generator.generate_synthetic_sequence(5)
    print(f"生成了 {len(synthetic_sequence)} 个合成样本")
    
    # 批处理
    batch_data = processor.batch_process(synthetic_sequence)
    print("批处理数据形状:")
    for key, value in batch_data.items():
        print(f"  {key}: {value.shape}")
