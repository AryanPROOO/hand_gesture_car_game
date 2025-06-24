import pygame
import sys
import cv2
import numpy as np
import mediapipe as mp
import random
import threading
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize PyGame
pygame.init()
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gesture Controlled Car Racing")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
GREEN = (50, 200, 50)
BLUE = (50, 150, 255)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)
ORANGE = (255, 150, 50)
PURPLE = (180, 50, 230)
ROAD_COLOR = (40, 40, 40)
LINE_COLOR = (200, 200, 0)
CHECKERED = (0, 0, 0)

# Road settings
ROAD_WIDTH = 600
ROAD_X = (WIDTH - ROAD_WIDTH) // 2
LANE_WIDTH = ROAD_WIDTH // 3
LINE_SPACING = 50
LINE_WIDTH = 10

# Car settings
PLAYER_WIDTH, PLAYER_HEIGHT = 50, 80
player_x = ROAD_X + ROAD_WIDTH // 2 - PLAYER_WIDTH // 2
player_y = HEIGHT - 150
player_speed = 0
player_lateral_speed = 0
MAX_FORWARD_SPEED = 10
MAX_LATERAL_SPEED = 8
ACCELERATION = 0.2
DECELERATION = 0.3
LATERAL_ACCEL = 0.4

# Game state
score = 0
game_active = True
race_started = False
winner_state = False
race_distance = 0
TOTAL_RACE_DISTANCE = 5000
font = pygame.font.SysFont(None, 36)
small_font = pygame.font.SysFont(None, 24)

# Obstacles
obstacles = []
OBSTACLE_SPEED = 5
OBSTACLE_SPAWN_DELAY = 45
obstacle_spawn_timer = 0
OBSTACLE_COLORS = [RED, PURPLE, ORANGE, (0, 200, 200)]

# Camera and gesture detection globals
current_gesture = "palm"
gesture_lock = threading.Lock()
camera_running = True

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Error: Could not open camera")
    sys.exit()

def detect_gesture(hand_landmarks):
    """Detect gestures using hand landmarks"""
    # Landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    
    # Get landmark positions
    landmarks = hand_landmarks.landmark
    
    # Check for fist (all fingers closed)
    fist = True
    for tip in [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        if landmarks[tip].y < landmarks[tip-2].y:  # Finger is open
            fist = False
    
    # Check for palm (all fingers open)
    palm = True
    for tip in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        if landmarks[tip].y > landmarks[tip-2].y:  # Finger is closed
            palm = False
    
    # Check for OK sign (thumb and index finger touching)
    thumb_tip = np.array([landmarks[THUMB_TIP].x, landmarks[THUMB_TIP].y])
    index_tip = np.array([landmarks[INDEX_TIP].x, landmarks[INDEX_TIP].y])
    distance = np.linalg.norm(thumb_tip - index_tip)
    ok_sign = distance < 0.05  # Threshold for touching
    
    # Check hand orientation for left/right
    if landmarks[WRIST].x < landmarks[MIDDLE_TIP].x:
        hand_direction = 'right'
    else:
        hand_direction = 'left'
    
    # Priority: OK > fist > palm > direction
    if ok_sign:
        return "ok"
    elif fist:
        return "fist"
    elif palm:
        return "palm"
    else:
        return hand_direction

def camera_thread():
    """Thread function for parallel camera window"""
    global current_gesture, camera_running
    
    # Create camera window
    cv2.namedWindow('Live Camera - Gesture Control', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Live Camera - Gesture Control', 480, 360)
    
    # Position camera window to the right of game window
    cv2.moveWindow('Live Camera - Gesture Control', 1020, 50)
    
    print("📹 Camera window opened parallel to game!")
    print("   - Position your hand in front of the camera")
    print("   - Make clear gestures for best recognition")
    
    while camera_running:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame for gesture detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        gesture = "palm"  # Default gesture
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on camera feed
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = detect_gesture(hand_landmarks)
                break
        
        # Update global gesture with thread safety
        with gesture_lock:
            current_gesture = gesture
        
        # Add text overlay showing current gesture
        h, w = frame.shape[:2]
        
        # Background rectangle for text
        cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 80), (0, 255, 0), 2)
        
        # Display current gesture
        cv2.putText(frame, f"Current Gesture: {gesture.upper()}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display instructions
        cv2.putText(frame, "Make gestures to control the car!", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add gesture guide
        instructions = [
            "CONTROLS:",
            "✊ Fist = Accelerate",
            "✋ Palm = Brake", 
            "👈 Left = Turn Left",
            "👉 Right = Turn Right",
            "👌 OK = Win at finish"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = h - 140 + (i * 22)
            cv2.putText(frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Add status indicator
        color = (0, 255, 0) if gesture != "palm" else (100, 100, 100)
        cv2.circle(frame, (w - 30, 30), 15, color, -1)
        cv2.putText(frame, "ACTIVE" if gesture != "palm" else "IDLE", 
                   (w - 80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display the camera frame
        cv2.imshow('Live Camera - Gesture Control', frame)
        
        # Check for window close or 'q' key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty('Live Camera - Gesture Control', cv2.WND_PROP_VISIBLE) < 1:
            camera_running = False
            break
    
    cv2.destroyAllWindows()

def draw_road(road_offset):
    """Draw the road with animated markings"""
    # Road surface
    pygame.draw.rect(screen, ROAD_COLOR, (ROAD_X, 0, ROAD_WIDTH, HEIGHT))
    
    # Animated lane markings
    for i in range(1, 3):
        lane_x = ROAD_X + i * LANE_WIDTH
        for y in range(-LINE_SPACING, HEIGHT, LINE_SPACING):
            line_y = (y + road_offset) % (HEIGHT + LINE_SPACING)
            pygame.draw.rect(screen, LINE_COLOR, (lane_x - LINE_WIDTH//2, line_y, LINE_WIDTH, LINE_SPACING//2))

def draw_player_car(x, y):
    """Draw the player's car"""
    # Car body
    pygame.draw.rect(screen, BLUE, (x, y, PLAYER_WIDTH, PLAYER_HEIGHT))
    pygame.draw.rect(screen, (30, 30, 100), (x, y, PLAYER_WIDTH, PLAYER_HEIGHT), 2)
    
    # Windows
    pygame.draw.rect(screen, (150, 220, 255), (x+5, y+10, PLAYER_WIDTH-10, 20))
    pygame.draw.rect(screen, (150, 220, 255), (x+5, y+40, PLAYER_WIDTH-10, 20))
    
    # Wheels
    pygame.draw.circle(screen, BLACK, (x+10, y+PLAYER_HEIGHT-10), 8)
    pygame.draw.circle(screen, BLACK, (x+PLAYER_WIDTH-10, y+PLAYER_HEIGHT-10), 8)
    pygame.draw.circle(screen, BLACK, (x+10, y+10), 8)
    pygame.draw.circle(screen, BLACK, (x+PLAYER_WIDTH-10, y+10), 8)

def draw_obstacle_car(x, y, color):
    """Draw an obstacle car"""
    # Car body
    pygame.draw.rect(screen, color, (x, y, PLAYER_WIDTH, PLAYER_HEIGHT))
    pygame.draw.rect(screen, (30, 30, 30), (x, y, PLAYER_WIDTH, PLAYER_HEIGHT), 2)
    
    # Windows
    pygame.draw.rect(screen, (200, 240, 255), (x+5, y+10, PLAYER_WIDTH-10, 20))
    
    # Wheels
    pygame.draw.circle(screen, BLACK, (x+10, y+PLAYER_HEIGHT-10), 8)
    pygame.draw.circle(screen, BLACK, (x+PLAYER_WIDTH-10, y+PLAYER_HEIGHT-10), 8)
    pygame.draw.circle(screen, BLACK, (x+10, y+10), 8)
    pygame.draw.circle(screen, BLACK, (x+PLAYER_WIDTH-10, y+10), 8)

def spawn_obstacle():
    """Create a new obstacle car"""
    lane = random.randint(0, 2)
    obstacle_x = ROAD_X + lane * LANE_WIDTH + (LANE_WIDTH - PLAYER_WIDTH) // 2
    color = random.choice(OBSTACLE_COLORS)
    return [obstacle_x, -PLAYER_HEIGHT, lane, color]

def draw_progress_bar():
    """Draw race progress bar"""
    # Background
    pygame.draw.rect(screen, (60, 60, 60), (WIDTH//2 - 150, 20, 300, 20))
    
    # Progress fill
    progress = min(1.0, race_distance / TOTAL_RACE_DISTANCE)
    progress_width = 296 * progress
    pygame.draw.rect(screen, GREEN, (WIDTH//2 - 148, 22, progress_width, 16))
    
    # Labels
    start_text = small_font.render("START", True, WHITE)
    finish_text = small_font.render("FINISH", True, WHITE)
    screen.blit(start_text, (WIDTH//2 - 170, 45))
    screen.blit(finish_text, (WIDTH//2 + 130, 45))
    
    # Percentage text
    percent_text = small_font.render(f"{int(progress*100)}%", True, WHITE)
    screen.blit(percent_text, (WIDTH//2 - 15, 45))

def show_text(text, position, color=WHITE):
    """Display text on screen"""
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)

# Start camera thread
print("🚀 Starting gesture-controlled car racing game...")
print("   - Main window: Racing game")
print("   - Side window: Live camera feed")

camera_thread_obj = threading.Thread(target=camera_thread, daemon=True)
camera_thread_obj.start()

# Give camera thread time to initialize
time.sleep(1)

# Main game loop
road_offset = 0

while True:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            camera_running = False
            cap.release()
            pygame.quit()
            sys.exit()
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r and not game_active:
                # Reset game state
                player_x = ROAD_X + ROAD_WIDTH // 2 - PLAYER_WIDTH // 2
                player_speed = 0
                player_lateral_speed = 0
                obstacles = []
                score = 0
                race_distance = 0
                game_active = True
                race_started = False
                winner_state = False
                road_offset = 0
                
            if event.key == pygame.K_q:
                camera_running = False
                cap.release()
                pygame.quit()
                sys.exit()
                
            if event.key == pygame.K_SPACE and not race_started:
                race_started = True
    
    # Get current gesture from camera thread
    with gesture_lock:
        gesture = current_gesture
    
    # Game logic when active
    if game_active and race_started:
        # Gesture-based controls
        if gesture == 'fist':
            player_speed = min(player_speed + ACCELERATION, MAX_FORWARD_SPEED)
        elif gesture == 'palm':
            player_speed = max(player_speed - DECELERATION, 0)
            
        if gesture == 'left':
            player_lateral_speed = max(player_lateral_speed - LATERAL_ACCEL, -MAX_LATERAL_SPEED)
        elif gesture == 'right':
            player_lateral_speed = min(player_lateral_speed + LATERAL_ACCEL, MAX_LATERAL_SPEED)
        else:
            # Gradually reduce lateral movement
            player_lateral_speed *= 0.9
            if abs(player_lateral_speed) < 0.1:
                player_lateral_speed = 0
        
        # Update player position (only horizontal)
        player_x += player_lateral_speed
        player_x = max(ROAD_X, min(player_x, ROAD_X + ROAD_WIDTH - PLAYER_WIDTH))
        
        # Update race progress
        race_distance += player_speed
        
        # Update road animation
        road_offset = (road_offset + player_speed) % LINE_SPACING
        
        # Spawn obstacles
        obstacle_spawn_timer += 1
        if obstacle_spawn_timer >= OBSTACLE_SPAWN_DELAY:
            obstacles.append(spawn_obstacle())
            obstacle_spawn_timer = 0
        
        # Update obstacles (move downward)
        for obstacle in obstacles[:]:
            obstacle[1] += OBSTACLE_SPEED + player_speed
            
            # Remove off-screen obstacles and update score
            if obstacle[1] > HEIGHT:
                obstacles.remove(obstacle)
                score += 1
        
        # Collision detection
        player_rect = pygame.Rect(player_x, player_y, PLAYER_WIDTH, PLAYER_HEIGHT)
        for obstacle in obstacles:
            obs_x, obs_y, _, _ = obstacle
            obs_rect = pygame.Rect(obs_x, obs_y, PLAYER_WIDTH, PLAYER_HEIGHT)
            
            if player_rect.colliderect(obs_rect):
                game_active = False
        
        # Win condition
        if race_distance >= TOTAL_RACE_DISTANCE:
            winner_state = True
            game_active = False
    
    # Rendering
    screen.fill(GREEN)  # Grass background
    
    # Draw road with animated markings
    draw_road(road_offset)
    
    # Draw start/finish lines (moving upward)
    start_line_y = HEIGHT - 100 - (race_distance % (HEIGHT - 200))
    pygame.draw.rect(screen, WHITE, (ROAD_X, start_line_y, ROAD_WIDTH, 5))
    
    finish_line_y = HEIGHT - 100 - ((race_distance + TOTAL_RACE_DISTANCE) % (HEIGHT - 200))
    pygame.draw.rect(screen, CHECKERED, (ROAD_X, finish_line_y, ROAD_WIDTH, 5))
    
    # Draw player car (fixed vertical position)
    draw_player_car(player_x, player_y)
    
    # Draw obstacles
    for obstacle in obstacles:
        obs_x, obs_y, _, color = obstacle
        draw_obstacle_car(obs_x, obs_y, color)
    
    # Draw progress bar
    draw_progress_bar()
    
    # Display game info
    show_text(f"Distance: {int(race_distance)}/{TOTAL_RACE_DISTANCE}", (20, 20))
    show_text(f"Speed: {player_speed:.1f}", (20, 60))
    show_text(f"Gesture: {gesture.upper()}", (20, 100))
    show_text(f"Score: {score}", (20, 140))
    show_text("Press Q to Quit", (WIDTH - 200, 20))
    
    # Camera status
    show_text("Camera: Active", (WIDTH - 200, 60), GREEN if camera_running else RED)
    show_text("Check side window", (WIDTH - 200, 100))
    
    # Start message
    if not race_started:
        start_msg = font.render("Press SPACE to Start Race", True, YELLOW)
        screen.blit(start_msg, (WIDTH//2 - 150, HEIGHT//2))
        
        cam_msg = small_font.render("Use gestures in camera window to control", True, WHITE)
        screen.blit(cam_msg, (WIDTH//2 - 150, HEIGHT//2 + 40))
    
    # Game over messages
    if not game_active:
        if winner_state:
            win_text = font.render("YOU WIN! Press R to restart", True, YELLOW)
            screen.blit(win_text, (WIDTH//2 - 180, HEIGHT//2))
        else:
            lose_text = font.render("GAME OVER! Press R to restart", True, RED)
            screen.blit(lose_text, (WIDTH//2 - 180, HEIGHT//2))
    
    pygame.display.flip()
    clock.tick(60)

# Cleanup
camera_running = False
cap.release()
pygame.quit()
sys.exit()
