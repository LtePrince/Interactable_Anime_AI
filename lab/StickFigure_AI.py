import pygame
import math
import time
import random
import logging
import os

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('galgame_ai.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 初始化pygame
pygame.init()
logger.info("Pygame初始化完成")

# 设置窗口大小
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
BACKGROUND_COLOR = (240, 240, 250)
STICK_COLOR = (50, 50, 50)
HAPPY_COLOR = (50, 200, 50)
ANGRY_COLOR = (200, 50, 50)
NEUTRAL_COLOR = (100, 100, 100)

class StickFigureAI:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.emotion_value = 5.0  # 情绪值 (1-10): 1=愤怒, 10=开心
        self.last_interaction_time = time.time()
        
        # 动画相关
        self.animation_frame = 0
        self.bounce_offset = 0
        self.arm_angle = 0
        self.leg_angle = 0
        self.head_bob = 0
        
        # 火柴人尺寸
        self.head_radius = 25
        self.body_length = 80
        self.arm_length = 40
        self.leg_length = 50
        
        # 拖拽检测
        self.is_being_dragged = False
        self.drag_start_pos = None
        
    def get_emotion_state(self):
        """根据情绪值返回情绪状态"""
        if self.emotion_value <= 3:
            return "angry"
        elif self.emotion_value <= 7:
            return "neutral"
        else:
            return "happy"
    
    def get_emotion_color(self):
        """根据情绪状态返回颜色"""
        state = self.get_emotion_state()
        if state == "angry":
            return ANGRY_COLOR
        elif state == "happy":
            return HAPPY_COLOR
        else:
            return NEUTRAL_COLOR
    
    def update_emotion(self):
        """更新情绪值 - 自动回归到中性"""
        current_time = time.time()
        
        # 3秒后开始自动回归到5（中性）
        if current_time - self.last_interaction_time > 3.0:
            if self.emotion_value > 5:
                self.emotion_value = max(5.0, self.emotion_value - 0.02)
            elif self.emotion_value < 5:
                self.emotion_value = min(5.0, self.emotion_value + 0.02)
    
    def handle_click(self, pos):
        """处理鼠标点击 - 增加开心度"""
        # 检查是否点击在火柴人附近
        distance = math.sqrt((pos[0] - self.x)**2 + (pos[1] - self.y)**2)
        if distance < 100:  # 点击范围
            self.emotion_value = min(10.0, self.emotion_value + 0.5)
            self.last_interaction_time = time.time()
            return True
        return False
    
    def handle_drag_start(self, pos):
        """开始拖拽"""
        distance = math.sqrt((pos[0] - self.x)**2 + (pos[1] - self.y)**2)
        if distance < 100:
            self.is_being_dragged = True
            self.drag_start_pos = pos
            return True
        return False
    
    def handle_drag(self, pos):
        """处理拖拽 - 增加愤怒度"""
        if self.is_being_dragged:
            self.emotion_value = max(1.0, self.emotion_value - 0.03)
            self.last_interaction_time = time.time()
    
    def handle_drag_end(self):
        """结束拖拽"""
        self.is_being_dragged = False
        self.drag_start_pos = None
    
    def update_animation(self):
        """更新动画帧"""
        self.animation_frame += 1
        
        # 根据情绪调整动画
        state = self.get_emotion_state()
        
        if state == "happy":
            # 开心时：跳跃动画
            self.bounce_offset = math.sin(self.animation_frame * 0.3) * 10
            self.arm_angle = math.sin(self.animation_frame * 0.2) * 20
            self.head_bob = math.sin(self.animation_frame * 0.4) * 5
        elif state == "angry":
            # 愤怒时：颤抖动画
            self.bounce_offset = random.randint(-3, 3)
            self.arm_angle = random.randint(-15, 15)
            self.head_bob = random.randint(-3, 3)
        else:
            # 中性时：缓慢摆动
            self.bounce_offset = math.sin(self.animation_frame * 0.1) * 3
            self.arm_angle = math.sin(self.animation_frame * 0.15) * 10
            self.head_bob = math.sin(self.animation_frame * 0.1) * 2
        
        # 腿部动画
        self.leg_angle = math.sin(self.animation_frame * 0.2) * 15
    
    def draw(self, screen):
        """绘制火柴人"""
        color = self.get_emotion_color()
        
        # 计算当前位置（包括动画偏移）
        current_x = self.x
        current_y = self.y + self.bounce_offset
        head_y = current_y - self.body_length - self.head_radius + self.head_bob
        
        # 绘制头部
        pygame.draw.circle(screen, color, (int(current_x), int(head_y)), self.head_radius, 3)
        
        # 绘制眼睛
        eye_size = 3
        if self.get_emotion_state() == "angry":
            # 愤怒的眼睛（斜线）
            pygame.draw.line(screen, color, 
                           (current_x - 10, head_y - 5), 
                           (current_x - 5, head_y + 5), 3)
            pygame.draw.line(screen, color, 
                           (current_x + 5, head_y - 5), 
                           (current_x + 10, head_y + 5), 3)
        else:
            # 正常眼睛
            pygame.draw.circle(screen, color, 
                             (int(current_x - 8), int(head_y - 5)), eye_size)
            pygame.draw.circle(screen, color, 
                             (int(current_x + 8), int(head_y - 5)), eye_size)
        
        # 绘制嘴巴
        if self.get_emotion_state() == "happy":
            # 笑脸
            pygame.draw.arc(screen, color, 
                          (current_x - 10, head_y, 20, 15), 
                          0, math.pi, 3)
        elif self.get_emotion_state() == "angry":
            # 愤怒的嘴巴
            pygame.draw.arc(screen, color, 
                          (current_x - 10, head_y + 5, 20, 15), 
                          math.pi, 2 * math.pi, 3)
        else:
            # 中性嘴巴
            pygame.draw.line(screen, color, 
                           (current_x - 8, head_y + 8), 
                           (current_x + 8, head_y + 8), 3)
        
        # 绘制身体
        body_top = head_y + self.head_radius
        body_bottom = body_top + self.body_length
        pygame.draw.line(screen, color, 
                        (current_x, body_top), 
                        (current_x, body_bottom), 5)
        
        # 绘制手臂
        arm_y = body_top + 20
        left_arm_x = current_x - self.arm_length * math.cos(math.radians(self.arm_angle))
        left_arm_y = arm_y + self.arm_length * math.sin(math.radians(self.arm_angle))
        right_arm_x = current_x + self.arm_length * math.cos(math.radians(self.arm_angle))
        right_arm_y = arm_y + self.arm_length * math.sin(math.radians(self.arm_angle))
        
        pygame.draw.line(screen, color, 
                        (current_x, arm_y), 
                        (left_arm_x, left_arm_y), 4)
        pygame.draw.line(screen, color, 
                        (current_x, arm_y), 
                        (right_arm_x, right_arm_y), 4)
        
        # 绘制腿部
        left_leg_x = current_x - self.leg_length * math.sin(math.radians(self.leg_angle)) / 2
        left_leg_y = body_bottom + self.leg_length * math.cos(math.radians(self.leg_angle))
        right_leg_x = current_x + self.leg_length * math.sin(math.radians(self.leg_angle)) / 2
        right_leg_y = body_bottom + self.leg_length * math.cos(math.radians(self.leg_angle))
        
        pygame.draw.line(screen, color, 
                        (current_x, body_bottom), 
                        (left_leg_x, left_leg_y), 4)
        pygame.draw.line(screen, color, 
                        (current_x, body_bottom), 
                        (right_leg_x, right_leg_y), 4)

def draw_ui(screen, font, ai):
    """绘制用户界面"""
    # 标题
    title = font.render("Interactable AI", True, (50, 50, 50))
    screen.blit(title, (10, 10))
    
    # 情绪值显示
    emotion_text = f"emotion value: {ai.emotion_value:.1f}/10"
    emotion_surface = font.render(emotion_text, True, ai.get_emotion_color())
    screen.blit(emotion_surface, (10, 50))
    
    # 情绪状态
    state_text = f"status: {ai.get_emotion_state()}"
    state_surface = font.render(state_text, True, ai.get_emotion_color())
    screen.blit(state_surface, (10, 80))
    
    # 情绪条
    bar_x, bar_y = 10, 110
    bar_width, bar_height = 200, 20
    
    # 背景条
    pygame.draw.rect(screen, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height))
    
    # 情绪填充
    fill_width = int((ai.emotion_value / 10.0) * bar_width)
    pygame.draw.rect(screen, ai.get_emotion_color(), (bar_x, bar_y, fill_width, bar_height))
    
    # 边框
    pygame.draw.rect(screen, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height), 2)
    
    # 操作说明
    instructions = [
        "Operator Instruction:",
        "• point = make it happy",
        "• drug = make it angry", 
        "• 3 seconds after interaction, it will return to neutral state",
        "• Quit with ESC key or close the window"
    ]
    
    for i, instruction in enumerate(instructions):
        color = (100, 100, 100) if i == 0 else (70, 70, 70)
        instruction_surface = font.render(instruction, True, color)
        screen.blit(instruction_surface, (10, 150 + i * 25))

def main():
    # 创建屏幕
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("可交互火柴人AI - 点击让它开心，拖拽让它愤怒")
    
    # 创建字体
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    # 创建AI角色
    ai = StickFigureAI(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    
    # 游戏循环
    clock = pygame.time.Clock()
    running = True
    
    print("🤖 火柴人AI已启动!")
    print("点击火柴人让它开心，拖拽让它愤怒!")
    
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    if ai.handle_click(event.pos):
                        print(f"点击! 开心度增加 - 当前情绪值: {ai.emotion_value:.1f}")
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
        draw_ui(screen, small_font, ai)
        
        # 更新显示
        pygame.display.flip()
        clock.tick(60)  # 60 FPS
    
    print("感谢使用火柴人AI! 再见! 👋")
    pygame.quit()

if __name__ == "__main__":
    main()
