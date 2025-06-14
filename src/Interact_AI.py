import pygame
import math
import time
import os
from utils.logger import get_logger, log_performance

# 初始化日志系统 - 使用项目根目录下的log文件夹
logger = get_logger("GalgameAI", log_dir="log", console_output=True)

# 初始化pygame
pygame.init()
logger.info("Pygame初始化完成")

# 设置窗口大小
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
BACKGROUND_COLOR = (240, 240, 250)

# 动画设置
JUMP_HEIGHT = 30
JUMP_DURATION = 20  # 帧数
FADE_DURATION = 15   # 淡入淡出持续帧数

class GalgameAI:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.base_y = y  # 基础Y坐标
        self.emotion_value = 5.0  # 情绪值 (1-10): 1=愤怒, 10=开心
        self.last_interaction_time = time.time()
        self.current_emotion_state = "neutral"
        self.previous_emotion_state = "neutral"
        
        # 图片尺寸 - 必须在load_images之前定义
        self.image_width = 200
        self.image_height = 300
        
        # 加载图片
        self.load_images()
        
        # 动画相关
        self.is_jumping = False
        self.jump_frame = 0
        self.current_y_offset = 0
        self.image_alpha = 255
        self.is_transitioning = False
        self.transition_frame = 0
        
        logger.info(f"Galgame AI初始化完成，位置: ({x}, {y})")
        
    def load_images(self):
        """加载情绪图片 - 保持原始比例"""
        self.images = {}
        self.actual_image_sizes = {}  # 存储实际图片尺寸
        image_files = {
            'angry': 'data/angry.png',
            'neutral': 'data/neutral.png', 
            'happy': 'data/happy.png'
        }
        
        # 设置最大显示尺寸
        max_width = 300
        max_height = 400
        
        try:
            for emotion, filepath in image_files.items():
                if os.path.exists(filepath):
                    original_image = pygame.image.load(filepath)
                    original_width, original_height = original_image.get_size()
                    
                    # 计算保持比例的缩放
                    width_ratio = max_width / original_width
                    height_ratio = max_height / original_height
                    scale_ratio = min(width_ratio, height_ratio)  # 选择较小的比例确保图片完全显示
                    
                    new_width = int(original_width * scale_ratio)
                    new_height = int(original_height * scale_ratio)
                    
                    # 缩放图片保持比例
                    scaled_image = pygame.transform.scale(original_image, (new_width, new_height))
                    self.images[emotion] = scaled_image
                    self.actual_image_sizes[emotion] = (new_width, new_height)
                    
                    logger.info(f"成功加载{emotion}情绪图片: {filepath}, 原始尺寸: {original_width}x{original_height}, 缩放后: {new_width}x{new_height}")
                else:
                    logger.error(f"图片文件不存在: {filepath}")
                    # 创建默认颜色矩形作为备用
                    colors = {'angry': (200, 50, 50), 'neutral': (100, 100, 100), 'happy': (50, 200, 50)}
                    image = pygame.Surface((self.image_width, self.image_height))
                    image.fill(colors[emotion])
                    self.images[emotion] = image
                    self.actual_image_sizes[emotion] = (self.image_width, self.image_height)
                    logger.warning(f"使用默认颜色替代{emotion}图片")
                    
        except Exception as e:
            logger.error(f"加载图片时发生错误: {e}")
            # 创建默认图片
            for emotion in ['angry', 'neutral', 'happy']:
                image = pygame.Surface((self.image_width, self.image_height))
                image.fill((100, 100, 100))
                self.images[emotion] = image
                self.actual_image_sizes[emotion] = (self.image_width, self.image_height)
            logger.warning("使用默认灰色图片作为所有情绪状态")
    
    def get_emotion_state(self):
        """根据情绪值返回情绪状态"""
        if self.emotion_value <= 3:
            return "angry"
        elif self.emotion_value <= 7:
            return "neutral"
        else:
            return "happy"
    
    @log_performance("GalgameAI")
    def update_emotion(self):
        """更新情绪值 - 自动回归到中性"""
        current_time = time.time()
        
        # 检查情绪状态是否改变
        new_state = self.get_emotion_state()
        if new_state != self.current_emotion_state:
            old_emotion = self.emotion_value
            old_state = self.current_emotion_state
            self.previous_emotion_state = self.current_emotion_state
            self.current_emotion_state = new_state
            self.start_transition()
            # 使用专用的情绪变化日志方法
            logger.log_emotion_change(old_emotion, self.emotion_value, old_state, new_state)
        
        # 3秒后开始自动回归到5（中性）
        if current_time - self.last_interaction_time > 3.0:
            old_value = self.emotion_value
            if self.emotion_value > 5:
                self.emotion_value = max(5.0, self.emotion_value - 0.02)
            elif self.emotion_value < 5:
                self.emotion_value = min(5.0, self.emotion_value + 0.02)
            
            # 如果情绪值发生了显著变化，记录日志
            if abs(old_value - self.emotion_value) > 0.1:
                if abs(old_value - self.emotion_value) > 0.5:
                    logger.info(f"情绪自动回归中: {old_value:.1f} -> {self.emotion_value:.1f}")
    
    def start_transition(self):
        """开始过渡动画"""
        self.is_jumping = True
        self.is_transitioning = True
        self.jump_frame = 0
        self.transition_frame = 0
        # 使用专用的动画事件日志方法
        logger.log_animation_event("过渡动画开始", {
            "emotion_state": self.current_emotion_state,
            "jump_height": JUMP_HEIGHT,
            "jump_duration": JUMP_DURATION,
            "fade_duration": FADE_DURATION
        })
    
    def handle_click(self, pos):
        """处理鼠标点击 - 增加开心度"""
        # 获取当前情绪状态的实际图片尺寸
        current_width, current_height = self.actual_image_sizes.get(self.current_emotion_state, (self.image_width, self.image_height))
        
        # 检查是否点击在图片区域
        img_rect = pygame.Rect(
            self.x - current_width // 2,
            self.y - current_height // 2 + self.current_y_offset,
            current_width,
            current_height
        )
        
        if img_rect.collidepoint(pos):
            old_value = self.emotion_value
            self.emotion_value = min(10.0, self.emotion_value + 0.5)
            self.last_interaction_time = time.time()
            # 使用专用的交互日志方法
            logger.log_interaction("点击", {
                "old_emotion": old_value,
                "new_emotion": self.emotion_value,
                "change": self.emotion_value - old_value,
                "position": pos
            })
            return True
        return False
    
    def handle_drag_start(self, pos):
        """开始拖拽"""
        # 获取当前情绪状态的实际图片尺寸
        current_width, current_height = self.actual_image_sizes.get(self.current_emotion_state, (self.image_width, self.image_height))
        
        img_rect = pygame.Rect(
            self.x - current_width // 2,
            self.y - current_height // 2 + self.current_y_offset,
            current_width,
            current_height
        )
        
        if img_rect.collidepoint(pos):
            self.is_being_dragged = True
            logger.info("开始拖拽操作")
            return True
        return False
    
    def handle_drag(self, pos):
        """处理拖拽 - 增加愤怒度"""
        if hasattr(self, 'is_being_dragged') and self.is_being_dragged:
            old_value = self.emotion_value
            self.emotion_value = max(1.0, self.emotion_value - 0.03)
            self.last_interaction_time = time.time()
            # 每秒记录一次拖拽日志，避免日志过多
            if int(time.time()) != getattr(self, 'last_drag_log_time', 0):
                logger.info(f"拖拽中，愤怒度增加: {self.emotion_value:.1f}")
                self.last_drag_log_time = int(time.time())
    
    def handle_drag_end(self):
        """结束拖拽"""
        if hasattr(self, 'is_being_dragged'):
            self.is_being_dragged = False
            logger.info("拖拽操作结束")
    
    def update_animation(self):
        """更新动画"""
        # 跳跃动画
        if self.is_jumping:
            self.jump_frame += 1
            
            # 使用sin函数创建平滑的跳跃动画
            progress = self.jump_frame / JUMP_DURATION
            if progress <= 1.0:
                # 跳跃曲线 (抛物线)
                self.current_y_offset = -JUMP_HEIGHT * math.sin(progress * math.pi)
            else:
                self.is_jumping = False
                self.current_y_offset = 0
                self.jump_frame = 0
        
        # 过渡动画（淡入淡出）
        if self.is_transitioning:
            self.transition_frame += 1
            
            if self.transition_frame <= FADE_DURATION:
                # 淡出阶段
                self.image_alpha = int(255 * (1 - self.transition_frame / FADE_DURATION))
            elif self.transition_frame <= FADE_DURATION * 2:
                # 淡入阶段
                fade_in_progress = (self.transition_frame - FADE_DURATION) / FADE_DURATION
                self.image_alpha = int(255 * fade_in_progress)
            else:
                # 动画结束
                self.is_transitioning = False
                self.image_alpha = 255
                self.transition_frame = 0
    
    @log_performance("GalgameAI")
    def draw(self, screen):
        """绘制AI角色"""
        # 获取当前图片和尺寸
        current_image = self.images[self.current_emotion_state].copy()
        current_width, current_height = self.actual_image_sizes.get(self.current_emotion_state, (self.image_width, self.image_height))
        
        # 应用透明度（用于过渡动画）
        if self.image_alpha < 255:
            current_image.set_alpha(self.image_alpha)
        
        # 计算绘制位置 - 使用实际图片尺寸
        draw_x = self.x - current_width // 2
        draw_y = self.y - current_height // 2 + self.current_y_offset
        
        # 绘制图片
        screen.blit(current_image, (draw_x, draw_y))
        
        # 如果正在过渡，可能需要绘制前一个状态的图片
        if self.is_transitioning and self.transition_frame <= FADE_DURATION:
            prev_image = self.images[self.previous_emotion_state].copy()
            prev_width, prev_height = self.actual_image_sizes.get(self.previous_emotion_state, (self.image_width, self.image_height))
            prev_alpha = int(255 * (1 - self.transition_frame / FADE_DURATION))
            prev_image.set_alpha(prev_alpha)
            
            # 前一个图片的绘制位置
            prev_draw_x = self.x - prev_width // 2
            prev_draw_y = self.y - prev_height // 2 + self.current_y_offset
            screen.blit(prev_image, (prev_draw_x, prev_draw_y))

def draw_ui(screen, font, ai):
    """绘制简化的用户界面 - 仅显示情绪值（英文）"""
    # 情绪值显示 - 大字体，放在屏幕顶部避免与图片冲突
    emotion_text = f"Emotion Level: {ai.emotion_value:.1f}/10"
    
    # 根据情绪值选择颜色
    if ai.emotion_value <= 3:
        color = (200, 50, 50)  # 愤怒红色
    elif ai.emotion_value <= 7:
        color = (100, 100, 100)  # 中性灰色
    else:
        color = (50, 200, 50)  # 开心绿色
    
    emotion_surface = font.render(emotion_text, True, color)
    text_rect = emotion_surface.get_rect()
    text_rect.centerx = WINDOW_WIDTH // 2
    text_rect.y = 20  # 放在更靠近顶部的位置
    screen.blit(emotion_surface, text_rect)
    
    # 情绪条 - 居中显示，放在屏幕顶部
    bar_width, bar_height = 300, 25
    bar_x = (WINDOW_WIDTH - bar_width) // 2
    bar_y = 60  # 调整位置避免与图片冲突
    
    # 背景条
    pygame.draw.rect(screen, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height))
    
    # 情绪填充
    fill_width = int((ai.emotion_value / 10.0) * bar_width)
    pygame.draw.rect(screen, color, (bar_x, bar_y, fill_width, bar_height))
    
    # 边框
    pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height), 3)
    
    # 情绪状态文字 - 英文显示
    emotion_states = {
        "angry": "ANGRY",
        "neutral": "NEUTRAL", 
        "happy": "HAPPY"
    }
    state_text = f"Status: {emotion_states.get(ai.current_emotion_state, ai.current_emotion_state.upper())}"
    state_surface = font.render(state_text, True, color)
    state_rect = state_surface.get_rect()
    state_rect.centerx = WINDOW_WIDTH // 2
    state_rect.y = 100  # 调整位置
    screen.blit(state_surface, state_rect)
    
    # 在屏幕底部显示简单操作提示（英文）
    small_font = pygame.font.Font(None, 24)
    tip_text = "Click to make happy | Drag to make angry | ESC to quit"
    tip_surface = small_font.render(tip_text, True, (120, 120, 120))
    tip_rect = tip_surface.get_rect()
    tip_rect.centerx = WINDOW_WIDTH // 2
    tip_rect.y = WINDOW_HEIGHT - 30
    screen.blit(tip_surface, tip_rect)

def main():
    # 创建屏幕
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Galgame风格AI - 点击增加开心度，拖拽增加愤怒度")
    
    # 创建字体
    font = pygame.font.Font(None, 48)  # 大字体用于情绪值显示
    
    # 创建AI角色
    ai = GalgameAI(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    
    # 游戏循环
    clock = pygame.time.Clock()
    running = True
    
    logger.info("🎮 Galgame风格AI已启动!")
    logger.info("操作说明: 点击角色增加开心度，拖拽角色增加愤怒度")
    
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                logger.info("用户关闭窗口，程序退出")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    logger.info("用户按ESC键，程序退出")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    if ai.handle_click(event.pos):
                        pass  # 日志已在handle_click中记录
                    # 开始拖拽检测
                    ai.handle_drag_start(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # 左键释放
                    ai.handle_drag_end()
            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:  # 左键按下并移动
                    ai.handle_drag(event.pos)
        
        # 更新AI状态
        ai.update_emotion()
        ai.update_animation()
        
        # 绘制
        screen.fill(BACKGROUND_COLOR)
        
        # 绘制AI
        ai.draw(screen)
        
        # 绘制UI
        draw_ui(screen, font, ai)
        
        # 更新显示
        pygame.display.flip()
        clock.tick(60)  # 60 FPS
    
    logger.info("感谢使用Galgame风格AI! 再见! 👋")
    pygame.quit()

if __name__ == "__main__":
    try:
        logger.info("程序启动...")
        logger.info(f"当前工作目录: {os.getcwd()}")
        logger.info(f"Python版本: {pygame.version.ver}")
        main()
    except ImportError as e:
        logger.error(f"导入模块错误: {e}")
        print("错误：请确保已安装pygame模块")
        print("安装命令：pip install pygame")
    except FileNotFoundError as e:
        logger.error(f"文件未找到错误: {e}")
        print("错误：请确保data目录和图片文件存在")
    except Exception as e:
        logger.error(f"程序运行时发生未知错误: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
    finally:
        pygame.quit()
        logger.info("Pygame已退出")
