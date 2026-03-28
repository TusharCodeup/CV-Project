"""
Animation Tools for AirCanvas Pro
Create moving drawings and flipbook animations
"""

import cv2
import numpy as np
import json
import time

class AnimationStudio:
    """Create animations from drawings"""
    
    def __init__(self, canvas_pro):
        self.canvas_pro = canvas_pro
        self.frames = []
        self.frame_delay = 0.1  # seconds
        self.looping = True
        
    def capture_frame(self):
        """Capture current frame for animation"""
        if len(self.frames) < 100:  # Limit frames
            self.frames.append(self.canvas_pro.canvas.copy())
            return True
        return False
    
    def play_animation(self):
        """Play captured animation"""
        if not self.frames:
            return
        
        window_name = "Animation Player"
        cv2.namedWindow(window_name)
        
        frame_idx = 0
        direction = 1
        
        while True:
            # Get frame
            frame = self.frames[frame_idx]
            
            # Display
            cv2.imshow(window_name, frame)
            
            # Handle controls
            key = cv2.waitKey(int(self.frame_delay * 1000)) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to pause/play
                cv2.waitKey(0)
            elif key == ord('r'):  # Reverse direction
                direction *= -1
            
            # Update frame index
            frame_idx += direction
            
            if frame_idx >= len(self.frames):
                if self.looping:
                    frame_idx = 0
                else:
                    break
            elif frame_idx < 0:
                if self.looping:
                    frame_idx = len(self.frames) - 1
                else:
                    break
        
        cv2.destroyWindow(window_name)
    
    def export_gif(self, filename):
        """Export animation as GIF"""
        try:
            import imageio
            
            writer = imageio.get_writer(filename, mode='I', fps=10)
            
            for frame in self.frames:
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            writer.close()
            print(f"🎬 GIF exported: {filename}")
            
        except ImportError:
            print("⚠️ Install imageio for GIF export: pip install imageio")
    
    def create_flipbook(self, num_frames=12):
        """Create flipbook animation"""
        print("📖 Creating flipbook animation...")
        
        # Capture frames over time
        for i in range(num_frames):
            self.capture_frame()
            time.sleep(0.1)
        
        print(f"✅ Captured {len(self.frames)} frames for flipbook")
    
    def add_motion(self, direction, speed=1):
        """Add motion to drawing"""
        if not self.frames:
            self.capture_frame()
        
        original = self.frames[0].copy()
        
        for i in range(1, len(self.frames)):
            # Create shifted copy
            shift_x = int(i * speed * direction[0])
            shift_y = int(i * speed * direction[1])
            
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            shifted = cv2.warpAffine(original, M, 
                                    (original.shape[1], original.shape[0]))
            
            self.frames[i] = shifted
    
    def add_particle_effect(self):
        """Add particle effect to animation"""
        for frame_idx, frame in enumerate(self.frames):
            particles = np.random.randint(0, 255, 
                                        (frame.shape[0], frame.shape[1]), 
                                        dtype=np.uint8)
            
            # Add particles as sparkles
            mask = particles > 250
            frame[mask] = [255, 255, 255]  # White sparkles
    
    def save_animation_data(self, filename):
        """Save animation data as JSON"""
        animation_data = {
            "num_frames": len(self.frames),
            "frame_delay": self.frame_delay,
            "looping": self.looping
        }
        
        with open(filename, 'w') as f:
            json.dump(animation_data, f, indent=2)
        
        print(f"💾 Animation data saved: {filename}")