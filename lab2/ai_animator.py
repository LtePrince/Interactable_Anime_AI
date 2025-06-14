import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. 定义超参数
EMOTION_DIM = 10  # 情绪状态范围 1-10
MOUSE_EVENT_DIM = 3 # 'a', 'b', 'none'
FRAME_SIZE = 5

# 情绪到嵌入的映射 (这里简单用一个线性层表示)
# 实际可以是一个更复杂的映射，或者直接使用one-hot编码
class EmotionEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmotionEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.embedding(x.float().unsqueeze(1)) # 确保输入是浮点数并增加一个维度

# 鼠标事件到嵌入的映射
class MouseEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MouseEmbedding, self).__init__()
        self.embedding = nn.Embedding(input_dim, output_dim) # input_dim是类别数量

    def forward(self, x):
        return self.embedding(x)


class ConditionalCRNN(nn.Module):
    def __init__(self, emotion_embedding_dim=8, mouse_embedding_dim=8,
                 frame_channels=1, frame_height=FRAME_SIZE, frame_width=FRAME_SIZE,
                 hidden_dim=128, output_channels=1):
        super(ConditionalCRNN, self).__init__()

        self.emotion_embedding = EmotionEmbedding(1, emotion_embedding_dim)
        self.mouse_embedding = MouseEmbedding(MOUSE_EVENT_DIM, mouse_embedding_dim)

        # CNN Encoder (处理5x5点阵)
        # 输入: (batch_size, channels, height, width)
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(frame_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 输出大小 (batch, 32, 2, 2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) # 输出大小 (batch, 64, 1, 1) - 如果是5x5，这里会是问题，需要调整
        )
        # 针对 5x5 输入，调整CNN Encoder输出扁平化的大小
        # 5x5 -> 3x3 (pool_size=2, ceil_mode=True) -> 1x1 (pool_size=2, ceil_mode=True)
        # out_size = (input_size - kernel_size + 2*padding) / stride + 1
        # pool_out_size = floor((input_size - pool_size) / pool_size) + 1  或者 ceil
        # 假设padding=0, kernel_size=3, stride=1, 5x5 -> 3x3, 再Maxpool2d(2) -> 1x1
        # 如果用padding=1, 5x5 -> 5x5, 再Maxpool2d(2) -> 2x2 或 3x3 (看ceil_mode)
        # 为了方便，我们假设Encoder的输出会展平到一个固定大小的向量
        self.encoded_frame_dim = 64 * (FRAME_SIZE // 4) * (FRAME_SIZE // 4) # 假设经过两次2x2池化
        if FRAME_SIZE == 5: # 对于5x5，经过两次MaxPool2d(2)后是1x1或2x2，这里需要精细计算
             self.encoded_frame_dim = 64 * ((FRAME_SIZE + 1) // 4) * ((FRAME_SIZE + 1) // 4) # 估算一下

        # LSTM/GRU layer
        self.hidden_dim = hidden_dim
        # 输入给RNN的维度是：编码帧维度 + 情绪嵌入维度 + 鼠标嵌入维度
        rnn_input_dim = self.encoded_frame_dim + emotion_embedding_dim + mouse_embedding_dim
        self.rnn = nn.GRU(rnn_input_dim, hidden_dim, batch_first=True)

        # CNN Decoder (生成5x5点阵)
        # 输入: (batch_size, hidden_dim) -> reshape to (batch_size, channels, H, W)
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 64, kernel_size=3, stride=1, padding=1), # 可以调整kernel_size和stride
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 上采样
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 上采样
            nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() # 输出0-1之间的值，表示点阵的亮度
        )
        # 确保解码器能正确生成5x5
        # 调整ConvTranspose2d的参数和Upsample，使其最终输出为5x5
        # 初始输入decoder的维度决定了其能生成的最大尺寸
        # 假设从hidden_dim开始，我们将它reshape成一个小的feature map
        self.decoder_initial_reshape_dim = 64
        self.decoder_initial_h = FRAME_SIZE // 4 + 1 if FRAME_SIZE % 4 != 0 else FRAME_SIZE // 4
        self.decoder_initial_w = FRAME_SIZE // 4 + 1 if FRAME_SIZE % 4 != 0 else FRAME_SIZE // 4
        self.cnn_decoder_initial_conv = nn.Sequential(
            nn.Linear(hidden_dim, self.decoder_initial_reshape_dim * self.decoder_initial_h * self.decoder_initial_w),
            nn.ReLU()
        )
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.decoder_initial_reshape_dim, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )


    def forward(self, prev_frame, mouse_event, emotion_state, hidden_state=None):
        # prev_frame: (batch_size, channels, H, W)
        # mouse_event: (batch_size,)  -> indices (0, 1, 2)
        # emotion_state: (batch_size,) -> scalars (1-10)

        # 1. 编码前一帧
        encoded_frame = self.cnn_encoder(prev_frame)
        encoded_frame = encoded_frame.view(encoded_frame.size(0), -1) # 展平

        # 2. 嵌入鼠标事件和情绪状态
        mouse_emb = self.mouse_embedding(mouse_event)
        emotion_emb = self.emotion_embedding(emotion_state)

        # 3. 拼接所有输入到RNN
        combined_input = torch.cat((encoded_frame, mouse_emb, emotion_emb), dim=1)
        combined_input = combined_input.unsqueeze(1) # 添加时间步维度 (batch, 1, input_dim)

        # 4. 经过RNN
        output, hidden_state = self.rnn(combined_input, hidden_state)

        # 5. 解码生成下一帧
        # output is (batch, 1, hidden_dim)
        decoded_input = self.cnn_decoder_initial_conv(output.squeeze(1)) # 移除时间步维度
        decoded_input = decoded_input.view(-1, self.decoder_initial_reshape_dim,
                                           self.decoder_initial_h, self.decoder_initial_w)
        next_frame = self.cnn_decoder(decoded_input)

        # 裁剪到5x5，因为转置卷积可能输出略大
        next_frame = next_frame[:, :, :FRAME_SIZE, :FRAME_SIZE]


        return next_frame, hidden_state

    # 初始化RNN的隐藏状态
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim) # (num_layers * num_directions, batch, hidden_size)
    
# 训练数据生成函数
def generate_data(num_samples, seq_len=10):
    data = []
    for _ in range(num_samples):
        # 随机生成初始帧
        initial_frame = np.random.randint(0, 2, size=(1, FRAME_SIZE, FRAME_SIZE)).astype(np.float32)
        frames = [initial_frame]
        emotions = [np.random.randint(1, 11)] # 初始情绪
        mouse_events = [np.random.randint(0, MOUSE_EVENT_DIM)] # 初始鼠标事件

        hidden = None # 初始隐藏状态

        for t in range(seq_len - 1):
            # 根据情绪生成目标帧 (哭脸/笑脸)
            current_emotion = emotions[-1]
            if current_emotion <= 3: # 愤怒 (1-3) -> 哭脸
                target_frame = np.array([
                    [0,0,0,0,0],
                    [0,1,0,1,0],
                    [0,0,0,0,0],
                    [0,1,1,1,0],
                    [1,0,0,0,1]
                ], dtype=np.float32).reshape(1, FRAME_SIZE, FRAME_SIZE)
            elif current_emotion >= 8: # 开心 (8-10) -> 笑脸
                target_frame = np.array([
                    [0,0,0,0,0],
                    [0,1,0,1,0],
                    [0,0,0,0,0],
                    [1,0,0,0,1],
                    [0,1,1,1,0]
                ], dtype=np.float32).reshape(1, FRAME_SIZE, FRAME_SIZE)
            else: # 平稳 (4-7) -> 随机或者保持
                target_frame = np.random.randint(0, 2, size=(1, FRAME_SIZE, FRAME_SIZE)).astype(np.float32)

            # 随机生成下一个鼠标事件和情绪
            next_mouse_event = np.random.randint(0, MOUSE_EVENT_DIM)
            next_emotion = np.clip(current_emotion + np.random.randint(-2, 3), 1, 10) # 情绪小幅度变化

            frames.append(target_frame)
            emotions.append(next_emotion)
            mouse_events.append(next_mouse_event)

        # 每个样本是一个序列：(帧序列, 鼠标事件序列, 情绪序列)
        data.append({
            'frames': np.array(frames), # (seq_len, 1, 5, 5)
            'mouse_events': np.array(mouse_events), # (seq_len,)
            'emotions': np.array(emotions) # (seq_len,)
        })
    return data

# 训练函数
def train_model(model, train_data, val_data, epochs=100, learning_rate=0.001, batch_size=16):
    criterion = nn.BCELoss() # 二分类交叉熵，因为输出是0-1的点阵
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(train_data)
        total_loss = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            if not batch:
                continue

            # 对齐序列长度 (可以填充，这里简单取最短)
            min_seq_len = min([len(s['frames']) for s in batch])
            if min_seq_len < 2: # 至少需要一个prev_frame和next_frame
                continue

            batch_prev_frames = []
            batch_next_frames = []
            batch_mouse_events = []
            batch_emotions = []

            for s in batch:
                batch_prev_frames.append(s['frames'][:min_seq_len-1])
                batch_next_frames.append(s['frames'][1:min_seq_len])
                batch_mouse_events.append(s['mouse_events'][:min_seq_len-1])
                batch_emotions.append(s['emotions'][:min_seq_len-1])

            prev_frames_tensor = torch.from_numpy(np.array(batch_prev_frames)).float().to(device)
            next_frames_tensor = torch.from_numpy(np.array(batch_next_frames)).float().to(device)
            mouse_events_tensor = torch.from_numpy(np.array(batch_mouse_events)).long().to(device)
            emotions_tensor = torch.from_numpy(np.array(batch_emotions)).float().to(device)


            optimizer.zero_grad()
            batch_loss = 0
            hidden = model.init_hidden(len(batch)).to(device)

            # 遍历序列进行训练 (teacher forcing)
            # 为了简化，这里假设每个序列的第一个元素是prev_frame，然后预测next_frame
            # 实际应用中可以利用RNN的seq2seq能力一次处理整个序列
            for t in range(min_seq_len - 1):
                current_prev_frame = prev_frames_tensor[:, t, :, :, :]
                current_mouse_event = mouse_events_tensor[:, t]
                current_emotion = emotions_tensor[:, t]
                current_next_frame_target = next_frames_tensor[:, t, :, :, :]

                output_frame, hidden = model(current_prev_frame, current_mouse_event, current_emotion, hidden)
                batch_loss += criterion(output_frame, current_next_frame_target)

            batch_loss /= (min_seq_len - 1)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / (len(train_data) / batch_size):.4f}")

        # 验证 (简化版，可以更完善)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i+batch_size]
                if not batch:
                    continue

                min_seq_len = min([len(s['frames']) for s in batch])
                if min_seq_len < 2:
                    continue

                batch_prev_frames = []
                batch_next_frames = []
                batch_mouse_events = []
                batch_emotions = []

                for s in batch:
                    batch_prev_frames.append(s['frames'][:min_seq_len-1])
                    batch_next_frames.append(s['frames'][1:min_seq_len])
                    batch_mouse_events.append(s['mouse_events'][:min_seq_len-1])
                    batch_emotions.append(s['emotions'][:min_seq_len-1])

                prev_frames_tensor = torch.from_numpy(np.array(batch_prev_frames)).float().to(device)
                next_frames_tensor = torch.from_numpy(np.array(batch_next_frames)).float().to(device)
                mouse_events_tensor = torch.from_numpy(np.array(batch_mouse_events)).long().to(device)
                emotions_tensor = torch.from_numpy(np.array(batch_emotions)).float().to(device)

                hidden = model.init_hidden(len(batch)).to(device)
                current_val_loss = 0
                for t in range(min_seq_len - 1):
                    current_prev_frame = prev_frames_tensor[:, t, :, :, :]
                    current_mouse_event = mouse_events_tensor[:, t]
                    current_emotion = emotions_tensor[:, t]
                    current_next_frame_target = next_frames_tensor[:, t, :, :, :]
                    output_frame, hidden = model(current_prev_frame, current_mouse_event, current_emotion, hidden)
                    current_val_loss += criterion(output_frame, current_next_frame_target)
                val_loss += current_val_loss.item() / (min_seq_len - 1)

        print(f"Validation Loss: {val_loss / (len(val_data) / batch_size):.4f}")

    print("Training finished!")

# ai_animator.py
# ... (上述模型和训练代码) ...

if __name__ == "__main__":
    # 数据生成
    num_train_samples = 2000
    num_val_samples = 400
    seq_len = 10 # 每个序列的帧数

    print("Generating training data...")
    train_data = generate_data(num_train_samples, seq_len)
    print("Generating validation data...")
    val_data = generate_data(num_val_samples, seq_len)
    print("Data generation complete.")

    # 模型实例化
    model = ConditionalCRNN()

    # 训练模型
    print("Starting training...")
    train_model(model, train_data, val_data, epochs=50, learning_rate=0.005, batch_size=32)

    # 保存模型 (可选)
    torch.save(model.state_dict(), "lab2/conditional_crnn_animator.pth")
    print("Model saved to conditional_crnn_animator.pth")

    # --- 可交互AI演示 (更新版) ---
    print("\n--- 可交互AI演示 ---")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    current_frame = np.array([
        [0,0,0,0,0],
        [0,0,1,0,0],
        [0,1,1,1,0],
        [0,0,1,0,0],
        [0,0,0,0,0]
    ], dtype=np.float32).reshape(1, 1, FRAME_SIZE, FRAME_SIZE) # 初始一个简单的形状

    current_emotion = 5 # 初始情绪平稳 (中立值)
    hidden_state = model.init_hidden(1).to(device)
    emotion_decay_rate = 0.5 # 情绪恢复速度，每次迭代恢复0.5个单位

    def display_frame(frame_array):
        print("\n----- Current Frame -----")
        for row in frame_array.squeeze().astype(int):
            print("".join(["*" if x == 1 else " " for x in row]))
        print("-------------------------")

    def map_emotion_to_str(emotion_val):
        if emotion_val <= 3: return "愤怒"
        elif emotion_val >= 8: return "开心"
        else: return "平稳"

    display_frame(current_frame)
    print(f"当前情绪: {map_emotion_to_str(current_emotion)} ({current_emotion:.1f}/10)") # 情绪显示一位小数

    while True:
        user_input = input("请输入鼠标事件 ('a'/'b'/'n' for none), 或 'q' 退出: ").lower()

        mouse_event_idx = 2 # 默认为None
        emotion_changed_by_input = False # 标记情绪是否因输入而改变

        if user_input == 'a':
            mouse_event_idx = 0
            current_emotion = max(1, current_emotion - 1) # 情绪减1，最低为1
            emotion_changed_by_input = True
        elif user_input == 'b':
            mouse_event_idx = 1
            current_emotion = min(10, current_emotion + 1) # 情绪加1，最高为10
            emotion_changed_by_input = True
        elif user_input == 'n':
            mouse_event_idx = 2 # 无鼠标事件
        elif user_input == 'q':
            break
        else:
            print("无效输入，请重新输入。")
            continue

        # 如果情绪不是由 'a'/'b' 直接改变，则缓慢恢复
        if not emotion_changed_by_input:
            if current_emotion > 5:
                current_emotion = max(5.0, current_emotion - emotion_decay_rate)
            elif current_emotion < 5:
                current_emotion = min(5.0, current_emotion + emotion_decay_rate)
            # 如果情绪已经非常接近5，直接设为5，避免浮点误差
            if abs(current_emotion - 5.0) < emotion_decay_rate/2:
                current_emotion = 5.0

        # 将Numpy数组转换为PyTorch张量
        prev_frame_tensor = torch.from_numpy(current_frame).float().to(device)
        mouse_event_tensor = torch.tensor([mouse_event_idx]).long().to(device)
        emotion_state_tensor = torch.tensor([current_emotion]).float().to(device)

        # 预测下一帧
        with torch.no_grad():
            next_frame_tensor, hidden_state = model(prev_frame_tensor, mouse_event_tensor, emotion_state_tensor, hidden_state)

        # 将预测的帧转换为Numpy数组并二值化显示 (大于0.5为1，否则为0)
        predicted_frame = (next_frame_tensor.cpu().numpy() > 0.5).astype(np.float32)

        current_frame = predicted_frame # 更新当前帧
        display_frame(current_frame)
        print(f"当前情绪: {map_emotion_to_str(current_emotion)} ({current_emotion:.1f}/10)")