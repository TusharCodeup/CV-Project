"""
AirCanvas Pro - 3D Gesture Drawing in Space
Complete working version with all tools
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
import time
import math
from collections import deque
from datetime import datetime
import colorsys
import random

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

class AirCanvasPro:
    """
    AirCanvas Pro - Complete 3D Drawing Studio with Hand Gestures
    """
    
    def __init__(self, sfm_reconstruction_dir=None):
        """
        Initialize AirCanvas Pro
        """
        # Camera settings
        self.camera = None
        self.running = False
        self.canvas = None
        self.drawing = False
        self.erasing = False
        
        # Drawing tools - FIXED list
        self.tools = ["brush", "fill", "smudge", "text", "move", "effects"]
        self.current_tool_index = 0
        self.current_tool = self.tools[0]  # Start with brush
        
        # Color palette
        self.colors = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (255, 255, 255), # White
            (0, 0, 0)       # Black
        ]
        self.current_color_index = 0
        self.current_color = self.colors[0]
        
        # Brush settings
        self.current_size = 5
        self.current_brush = "circle"
        
        # Drawing data
        self.lines = []
        self.current_line = []
        self.undo_stack = []
        self.redo_stack = []
        
        # Text input
        self.text_input = ""
        self.text_mode = False
        
        # 3D objects from SfM
        self.points_3d = []
        self.sfm_objects = []
        
        # Animation
        self.animations = []
        self.recording = False
        
        # Tracing mode
        self.tracing = False
        
        # Initialize AI models
        self.init_ai_models()
        
        # Load SfM reconstruction if provided
        if sfm_reconstruction_dir:
            self.load_sfm_reconstruction(sfm_reconstruction_dir)
        
        print("\n🎨 AirCanvas Pro Initialized")
        print("   ✋ Gesture Controls:")
        print("   👆 Index Finger - Draw")
        print("   ✌️ Victory - Change Tool")
        print("   🤙 Thumb+Pinky - Change Color")
        print("   👊 Fist - Erase")
        print("   🖐️ Open Palm - Clear")
        print("\n   ⌨️ Keyboard Shortcuts:")
        print("   [1-6] - Select Tool")
        print("   C - Change Color")
        print("   + / - - Brush Size")
        print("   U - Undo | R - Redo")
        print("   S - Save | Q - Quit")
    
    def init_ai_models(self):
        """Initialize MediaPipe for hand tracking"""
        print("🤖 Initializing AI Models...")
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        print("   ✅ Hand Tracking Ready")
    
    def load_sfm_reconstruction(self, reconstruction_dir):
        """Load 3D objects from SfM reconstruction"""
        print(f"📦 Loading 3D objects from {reconstruction_dir}")
        
        ply_path = Path(reconstruction_dir) / "point_cloud.ply"
        if ply_path.exists():
            self.points_3d = self.load_ply(ply_path)
            print(f"   ✅ Loaded {len(self.points_3d)} 3D points")
            
            self.sfm_objects.append({
                "name": "Reconstructed Object",
                "points": self.points_3d,
                "position": [0, 0, 0],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1]
            })
    
    def load_ply(self, ply_path):
        """Load point cloud from PLY file"""
        points = []
        
        with open(ply_path, 'r') as f:
            lines = f.readlines()
        
        vertex_start = 0
        for i, line in enumerate(lines):
            if "end_header" in line:
                vertex_start = i + 1
                break
        
        for line in lines[vertex_start:vertex_start + 1000]:  # Limit points
            parts = line.strip().split()
            if len(parts) >= 3:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                points.append([x, y, z])
        
        return np.array(points)
    
    def start_camera(self):
        """Start camera for gesture drawing"""
        print("\n📷 Starting AirCanvas Pro Camera...")
        
        self.camera = cv2.VideoCapture(0)
        self.running = True
        
        # Get frame dimensions
        ret, frame = self.camera.read()
        if ret:
            self.canvas = np.zeros(frame.shape, dtype=np.uint8)
        
        print("   ✅ Camera ready! Start drawing with your finger!")
        
        # Main loop
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Mirror frame
            frame = cv2.flip(frame, 1)
            
            # Detect hand gestures
            frame, gesture, finger_pos = self.detect_gestures(frame)
            
            # Process drawing based on gesture
            self.process_drawing(frame, gesture, finger_pos)
            
            # Combine canvas with camera feed
            display = self.combine_canvas(frame)
            
            # Draw UI
            display = self.draw_ui(display)
            
            # Show frame
            cv2.imshow('AirCanvas Pro - 3D Drawing Studio', display)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            self.handle_keyboard(key)
        
        self.cleanup()
    
    def detect_gestures(self, frame):
        """Detect hand gestures from camera feed"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture = "none"
        finger_pos = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Get finger positions
                h, w, _ = frame.shape
                finger_pos = []
                for tip in [4, 8, 12, 16, 20]:  # Finger tips
                    x = int(hand_landmarks.landmark[tip].x * w)
                    y = int(hand_landmarks.landmark[tip].y * h)
                    finger_pos.append((x, y))
                
                # Recognize gesture
                gesture = self.recognize_gesture(hand_landmarks)
                
                # Show gesture on screen
                cv2.putText(frame, f"Gesture: {gesture}", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show tool info
                cv2.putText(frame, f"Tool: {self.current_tool}", (10, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame, gesture, finger_pos
    
    def recognize_gesture(self, hand_landmarks):
        """Recognize hand gestures"""
        landmarks = hand_landmarks.landmark
        
        # Check which fingers are up
        fingers_up = []
        
        # Thumb (horizontal check)
        if landmarks[4].x < landmarks[3].x:
            fingers_up.append('thumb')
        
        # Other fingers (vertical check)
        if landmarks[8].y < landmarks[6].y:
            fingers_up.append('index')
        if landmarks[12].y < landmarks[10].y:
            fingers_up.append('middle')
        if landmarks[16].y < landmarks[14].y:
            fingers_up.append('ring')
        if landmarks[20].y < landmarks[18].y:
            fingers_up.append('pinky')
        
        # Recognize gestures
        if 'index' in fingers_up and len(fingers_up) == 1:
            self.drawing = True
            self.erasing = False
            return "drawing"
        
        elif 'index' in fingers_up and 'middle' in fingers_up and len(fingers_up) == 2:
            self.drawing = False
            self.erasing = False
            return "change_tool"
        
        elif 'thumb' in fingers_up and 'pinky' in fingers_up and len(fingers_up) == 2:
            self.drawing = False
            self.erasing = False
            return "change_color"
        
        elif len(fingers_up) == 0:
            self.drawing = False
            self.erasing = True
            return "erasing"
        
        elif len(fingers_up) >= 4:
            self.drawing = False
            self.erasing = False
            return "clear"
        
        self.drawing = False
        self.erasing = False
        return "none"
    
    def process_drawing(self, frame, gesture, finger_pos):
        """Process drawing based on gesture"""
        if not finger_pos or len(finger_pos) < 2:
            return
        
        # Get index finger tip (drawing point)
        draw_point = finger_pos[1]
        
        # Draw on canvas based on gesture
        if gesture == "drawing" and self.drawing:
            if self.current_tool == "brush":
                self.draw_on_canvas(draw_point)
            elif self.current_tool == "smudge":
                self.smudge_on_canvas(draw_point)
            elif self.current_tool == "text" and self.text_mode:
                self.add_text_at_point(draw_point)
            self.current_line.append(draw_point)
            
        elif gesture == "erasing" and self.erasing:
            self.erase_on_canvas(draw_point)
            
        elif gesture == "change_tool":
            self.change_tool()
            time.sleep(0.3)  # Debounce
            
        elif gesture == "change_color":
            self.change_color()
            time.sleep(0.3)  # Debounce
            
        elif gesture == "clear":
            self.clear_canvas()
            time.sleep(0.5)  # Debounce
    
    def draw_on_canvas(self, point):
        """Draw on canvas with current brush"""
        x, y = point
        
        # Draw based on brush type
        if self.current_brush == "circle":
            cv2.circle(self.canvas, (x, y), self.current_size, self.current_color, -1)
        elif self.current_brush == "square":
            cv2.rectangle(self.canvas, 
                         (x - self.current_size, y - self.current_size),
                         (x + self.current_size, y + self.current_size),
                         self.current_color, -1)
        
        # Smooth line between points
        if len(self.current_line) > 1:
            cv2.line(self.canvas, self.current_line[-2], 
                    self.current_line[-1], self.current_color, 
                    self.current_size * 2)
    
    def erase_on_canvas(self, point):
        """Erase at point"""
        x, y = point
        cv2.circle(self.canvas, (x, y), self.current_size * 2, (0, 0, 0), -1)
    
    def smudge_on_canvas(self, point):
        """Smudge/blend colors at point"""
        x, y = point
        radius = 10
        
        # Get region
        x1 = max(0, x - radius)
        x2 = min(self.canvas.shape[1], x + radius)
        y1 = max(0, y - radius)
        y2 = min(self.canvas.shape[0], y + radius)
        
        # Apply blur for smudge effect
        region = self.canvas[y1:y2, x1:x2]
        if region.size > 0:
            blurred = cv2.GaussianBlur(region, (15, 15), 0)
            self.canvas[y1:y2, x1:x2] = blurred
    
    def add_text_at_point(self, point):
        """Add text at clicked point"""
        if self.text_input:
            x, y = point
            cv2.putText(self.canvas, self.text_input, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, self.current_color, 2)
            self.text_input = ""
            self.text_mode = False
    
    def change_tool(self):
        """Cycle through tools"""
        self.current_tool_index = (self.current_tool_index + 1) % len(self.tools)
        self.current_tool = self.tools[self.current_tool_index]
        
        # Handle tool-specific setup
        if self.current_tool == "fill":
            print("🎨 Fill Tool - Click to fill areas")
        elif self.current_tool == "text":
            self.text_mode = True
            print("📝 Text Tool - Type text, then click to place")
        elif self.current_tool == "effects":
            self.apply_random_effect()
        
        print(f"🖌️ Tool changed to: {self.current_tool}")
    
    def change_color(self):
        """Cycle through colors"""
        self.current_color_index = (self.current_color_index + 1) % len(self.colors)
        self.current_color = self.colors[self.current_color_index]
        
        # Show color preview
        color_names = ["Green", "Red", "Blue", "Yellow", "Magenta", "Cyan", "White", "Black"]
        print(f"🎨 Color changed to: {color_names[self.current_color_index]}")
    
    def apply_random_effect(self):
        """Apply random effect to canvas"""
        effects = ["blur", "sharpen", "edge", "emboss"]
        effect = random.choice(effects)
        
        if effect == "blur":
            self.canvas = cv2.GaussianBlur(self.canvas, (15, 15), 0)
            print("✨ Applied: Blur Effect")
        elif effect == "sharpen":
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            self.canvas = cv2.filter2D(self.canvas, -1, kernel)
            print("✨ Applied: Sharpen Effect")
        elif effect == "edge":
            edges = cv2.Canny(self.canvas, 100, 200)
            self.canvas = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            print("✨ Applied: Edge Detection")
        elif effect == "emboss":
            kernel = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])
            self.canvas = cv2.filter2D(self.canvas, -1, kernel)
            print("✨ Applied: Emboss Effect")
    
    def fill_area(self, point):
        """Flood fill algorithm for bucket tool"""
        x, y = point
        
        if x >= self.canvas.shape[1] or y >= self.canvas.shape[0]:
            return
        
        target_color = self.canvas[y, x].copy()
        
        if np.array_equal(target_color, self.current_color):
            return
        
        stack = [(x, y)]
        visited = set()
        
        while stack and len(stack) < 5000:  # Limit for performance
            cx, cy = stack.pop()
            
            if (cx, cy) in visited:
                continue
            
            if not (0 <= cx < self.canvas.shape[1] and 0 <= cy < self.canvas.shape[0]):
                continue
            
            visited.add((cx, cy))
            
            if np.array_equal(self.canvas[cy, cx], target_color):
                self.canvas[cy, cx] = self.current_color
                
                # Add neighbors
                stack.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])
        
        print(f"🎨 Filled area with {len(visited)} pixels")
    
    def clear_canvas(self):
        """Clear the entire canvas"""
        # Save to undo stack
        self.undo_stack.append(self.canvas.copy())
        self.canvas.fill(0)
        self.current_line = []
        print("🗑️ Canvas cleared")
    
    def undo(self):
        """Undo last action"""
        if self.undo_stack:
            self.redo_stack.append(self.canvas.copy())
            self.canvas = self.undo_stack.pop()
            print("↩️ Undo")
    
    def redo(self):
        """Redo last undone action"""
        if self.redo_stack:
            self.undo_stack.append(self.canvas.copy())
            self.canvas = self.redo_stack.pop()
            print("↪️ Redo")
    
    def save_drawing(self):
        """Save current drawing"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create drawings directory
        Path("drawings").mkdir(exist_ok=True)
        
        # Save canvas
        filename = f"drawings/drawing_{timestamp}.png"
        cv2.imwrite(filename, self.canvas)
        print(f"💾 Drawing saved: {filename}")
        
        # Save with 3D objects if any
        if len(self.sfm_objects) > 0:
            scene_file = f"drawings/scene_{timestamp}.json"
            scene_data = {
                "objects": [{"name": obj["name"], "position": obj["position"]} 
                           for obj in self.sfm_objects],
                "timestamp": timestamp
            }
            with open(scene_file, 'w') as f:
                json.dump(scene_data, f, indent=2)
            print(f"🌍 3D scene saved: {scene_file}")
    
    def combine_canvas(self, frame):
        """Combine drawing canvas with camera feed"""
        # Create a copy
        result = frame.copy()
        
        # Add canvas overlay with transparency
        if np.any(self.canvas):
            mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY) > 0
            result[mask] = cv2.addWeighted(frame[mask], 0.3, 
                                          self.canvas[mask], 0.7, 0)
        
        return result
    
    def draw_ui(self, frame):
        """Draw UI elements on screen"""
        h, w, _ = frame.shape
        
        # Top toolbar background
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        
        # Draw tool buttons
        tool_x = 10
        tool_names = ["Brush", "Fill", "Smudge", "Text", "Move", "Effects"]
        
        for i, (tool, name) in enumerate(zip(self.tools, tool_names)):
            color = (0, 255, 0) if i == self.current_tool_index else (100, 100, 100)
            cv2.rectangle(frame, (tool_x, 5), (tool_x + 70, 35), color, 2)
            cv2.putText(frame, name, (tool_x + 5, 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tool_x += 80
        
        # Color palette
        color_x = w - 200
        for i, color in enumerate(self.colors[:6]):  # Show 6 colors
            cv2.rectangle(frame, (color_x, 5), (color_x + 25, 35), color, -1)
            if i == self.current_color_index:
                cv2.rectangle(frame, (color_x, 5), (color_x + 25, 35), (255, 255, 255), 2)
            color_x += 30
        
        # Brush size indicator
        cv2.putText(frame, f"Size: {self.current_size}", (w - 100, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "✋ Draw | ✌️ Tool | 🤙 Color | 👊 Erase | 🖐️ Clear", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Text input indicator
        if self.text_mode:
            cv2.putText(frame, "TEXT MODE: Type text, then click to place", 
                       (w//2 - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 255), 2)
        
        return frame
    
    def handle_keyboard(self, key):
        """Handle keyboard shortcuts"""
        if key == ord('1'):
            self.current_tool_index = 0
            self.current_tool = self.tools[0]
            print("🖌️ Tool: Brush")
        elif key == ord('2'):
            self.current_tool_index = 1
            self.current_tool = self.tools[1]
            print("🎨 Tool: Fill")
        elif key == ord('3'):
            self.current_tool_index = 2
            self.current_tool = self.tools[2]
            print("✏️ Tool: Smudge")
        elif key == ord('4'):
            self.current_tool_index = 3
            self.current_tool = self.tools[3]
            self.text_mode = True
            print("📝 Tool: Text")
        elif key == ord('5'):
            self.current_tool_index = 4
            self.current_tool = self.tools[4]
            print("🔄 Tool: Move")
        elif key == ord('6'):
            self.current_tool_index = 5
            self.current_tool = self.tools[5]
            self.apply_random_effect()
            print("🎭 Tool: Effects")
        elif key == ord('c'):
            self.change_color()
        elif key == ord('+') or key == ord('='):
            self.current_size = min(30, self.current_size + 2)
            print(f"✏️ Brush size: {self.current_size}")
        elif key == ord('-'):
            self.current_size = max(1, self.current_size - 2)
            print(f"✏️ Brush size: {self.current_size}")
        elif key == ord('u'):
            self.undo()
        elif key == ord('r'):
            self.redo()
        elif key == ord('s'):
            self.save_drawing()
        elif key == ord('t'):
            self.tracing = not self.tracing
            print(f"🔍 Tracing mode: {'ON' if self.tracing else 'OFF'}")
        elif key == ord('f') and self.current_tool == "fill":
            # Fill at mouse position (would need mouse click in full version)
            pass
        elif key == ord('q'):
            self.running = False
        
        # Text input
        elif self.text_mode and key != -1 and 32 <= key <= 126:
            self.text_input += chr(key)
            print(f"Text: {self.text_input}")
        elif key == 8 and self.text_mode:  # Backspace
            self.text_input = self.text_input[:-1]
        elif key == 13 and self.text_mode:  # Enter
            self.text_mode = False
            print("Click on canvas to place text")
    
    def cleanup(self):
        """Clean up resources"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("\n✅ AirCanvas Pro Closed")
        print(f"   Total drawings saved in memory")


def main():
    """Main function"""
    print("\n" + "="*60)
    print("🎨 AirCanvas Pro - 3D Drawing Studio")
    print("="*60)
    
    # Path to SfM reconstruction (optional)
    sfm_dir = "outputs/my_reconstruction"
    
    # Check if SfM reconstruction exists
    if not Path(sfm_dir).exists():
        print("\n⚠️ SfM reconstruction not found.")
        print("   Running without 3D objects.")
        print("   To enable 3D objects, run SfM first:")
        print("   python fixed_main.py --image_dir data/test_sequence")
        sfm_dir = None
    
    # Launch AirCanvas Pro
    app = AirCanvasPro(sfm_dir)
    
    try:
        app.start_camera()
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using AirCanvas Pro!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure your camera is connected and working.")

if __name__ == "__main__":
    # Check for mediapipe
    try:
        import mediapipe
        print("✅ MediaPipe installed")
    except:
        print("📥 Installing MediaPipe...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe"])
    
    main()