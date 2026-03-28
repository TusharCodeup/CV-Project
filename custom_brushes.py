"""
Custom Brushes and Textures Module
"""

import cv2
import numpy as np
import random

class BrushLibrary:
    """Collection of custom brushes and textures"""
    
    def __init__(self):
        self.brushes = {
            "pencil": self.pencil_brush,
            "marker": self.marker_brush,
            "watercolor": self.watercolor_brush,
            "spray": self.spray_brush,
            "calligraphy": self.calligraphy_brush,
            "stamp": self.stamp_brush,
            "gradient": self.gradient_brush,
            "pattern": self.pattern_brush
        }
        
        self.textures = self.load_textures()
    
    def pencil_brush(self, size, hardness=0.8):
        """Create pencil-like brush with texture"""
        brush = np.zeros((size*2, size*2), dtype=np.uint8)
        
        for i in range(size*2):
            for j in range(size*2):
                distance = np.sqrt((i-size)**2 + (j-size)**2)
                if distance < size:
                    # Add noise for pencil texture
                    noise = random.randint(0, 50)
                    value = int(255 * (1 - distance/size) * hardness + noise)
                    brush[i, j] = min(255, value)
        
        return brush
    
    def marker_brush(self, size):
        """Create marker brush with opacity variation"""
        brush = np.zeros((size*2, size*2), dtype=np.uint8)
        
        for i in range(size*2):
            for j in range(size*2):
                distance = np.sqrt((i-size)**2 + (j-size)**2)
                if distance < size:
                    value = int(255 * (1 - (distance/size)**0.5))
                    brush[i, j] = value
        
        return brush
    
    def watercolor_brush(self, size):
        """Create watercolor effect brush"""
        brush = np.zeros((size*2, size*2), dtype=np.uint8)
        
        for i in range(size*2):
            for j in range(size*2):
                distance = np.sqrt((i-size)**2 + (j-size)**2)
                if distance < size:
                    # Random splotchy pattern
                    splotch = random.random() * 100
                    value = int(255 * (1 - distance/size) * (0.7 + splotch/255))
                    brush[i, j] = min(255, value)
        
        return brush
    
    def spray_brush(self, size, density=0.3):
        """Create spray paint brush"""
        brush = np.zeros((size*2, size*2), dtype=np.uint8)
        
        for i in range(size*2):
            for j in range(size*2):
                distance = np.sqrt((i-size)**2 + (j-size)**2)
                if distance < size and random.random() < density:
                    value = int(255 * (1 - distance/size))
                    brush[i, j] = value
        
        return brush
    
    def calligraphy_brush(self, size):
        """Create calligraphy brush (angled)"""
        brush = np.zeros((size*2, size*2), dtype=np.uint8)
        angle = 45  # degrees
        
        for i in range(size*2):
            for j in range(size*2):
                # Transform coordinates based on angle
                x = (i-size) * np.cos(np.radians(angle)) - (j-size) * np.sin(np.radians(angle))
                y = (i-size) * np.sin(np.radians(angle)) + (j-size) * np.cos(np.radians(angle))
                
                if abs(x) < size/2 and abs(y) < size:
                    brush[i, j] = 255
        
        return brush
    
    def stamp_brush(self, size, pattern="star"):
        """Create stamp brushes"""
        brush = np.zeros((size*2, size*2), dtype=np.uint8)
        
        if pattern == "star":
            center = (size, size)
            points = []
            for i in range(5):
                angle = i * 72 * np.pi / 180
                r1 = size
                r2 = size // 2
                x1 = center[0] + r1 * np.cos(angle)
                y1 = center[1] + r1 * np.sin(angle)
                x2 = center[0] + r2 * np.cos(angle + 36 * np.pi / 180)
                y2 = center[1] + r2 * np.sin(angle + 36 * np.pi / 180)
                points.extend([(int(x1), int(y1)), (int(x2), int(y2))])
            
            cv2.fillPoly(brush, [np.array(points)], 255)
        
        elif pattern == "heart":
            # Heart shape formula
            for i in range(size*2):
                for j in range(size*2):
                    x = (i - size) / size
                    y = (j - size) / size
                    if (x*x + y*y - 1)**3 - x*x * y*y*y < 0:
                        brush[i, j] = 255
        
        return brush
    
    def gradient_brush(self, size, colors=None):
        """Create gradient brush"""
        if colors is None:
            colors = [(255,0,0), (0,255,0), (0,0,255)]
        
        brush = np.zeros((size*2, size*2, 3), dtype=np.uint8)
        
        for i in range(size*2):
            for j in range(size*2):
                distance = np.sqrt((i-size)**2 + (j-size)**2)
                if distance < size:
                    t = distance / size
                    idx = int(t * (len(colors) - 1))
                    if idx < len(colors) - 1:
                        t2 = t * (len(colors) - 1) - idx
                        color = tuple(int(c1 * (1-t2) + c2 * t2) 
                                    for c1, c2 in zip(colors[idx], colors[idx+1]))
                    else:
                        color = colors[-1]
                    brush[i, j] = color
        
        return brush
    
    def pattern_brush(self, size, pattern="dots"):
        """Create pattern brushes"""
        brush = np.zeros((size*2, size*2), dtype=np.uint8)
        
        if pattern == "dots":
            spacing = size // 3
            for i in range(0, size*2, spacing):
                for j in range(0, size*2, spacing):
                    cv2.circle(brush, (i, j), size//4, 255, -1)
        
        elif pattern == "grid":
            spacing = size // 2
            for i in range(0, size*2, spacing):
                cv2.line(brush, (i, 0), (i, size*2), 255, 2)
                cv2.line(brush, (0, i), (size*2, i), 255, 2)
        
        elif pattern == "crosshatch":
            spacing = size // 3
            for i in range(0, size*2, spacing):
                cv2.line(brush, (i, 0), (size*2, size*2-i), 255, 1)
                cv2.line(brush, (0, i), (size*2-i, size*2), 255, 1)
        
        return brush
    
    def load_textures(self):
        """Load texture patterns"""
        textures = {
            "canvas": self.create_canvas_texture,
            "paper": self.create_paper_texture,
            "wood": self.create_wood_texture,
            "marble": self.create_marble_texture
        }
        return textures
    
    def create_canvas_texture(self, size):
        """Create canvas texture"""
        texture = np.ones((size, size), dtype=np.uint8) * 200
        
        # Add noise
        noise = np.random.randint(0, 30, (size, size), dtype=np.uint8)
        texture = cv2.add(texture, noise)
        
        # Add grid pattern
        spacing = 20
        for i in range(0, size, spacing):
            cv2.line(texture, (i, 0), (i, size), 180, 1)
            cv2.line(texture, (0, i), (size, i), 180, 1)
        
        return texture
    
    def create_paper_texture(self, size):
        """Create paper texture"""
        texture = np.ones((size, size), dtype=np.uint8) * 250
        
        # Add subtle noise
        noise = np.random.randint(0, 20, (size, size), dtype=np.uint8)
        texture = cv2.subtract(texture, noise)
        
        # Add fibers
        for _ in range(100):
            x = random.randint(0, size)
            y = random.randint(0, size)
            cv2.line(texture, (x, y), (x+random.randint(5,15), y+random.randint(0,5)), 200, 1)
        
        return texture
    
    def create_wood_texture(self, size):
        """Create wood grain texture"""
        texture = np.zeros((size, size), dtype=np.uint8)
        
        for i in range(size):
            # Create wavy lines for grain
            wave = int(30 * np.sin(i * 0.1))
            value = 128 + wave
            texture[i, :] = value
        
        # Add noise
        noise = np.random.randint(0, 30, (size, size), dtype=np.uint8)
        texture = cv2.add(texture, noise)
        
        return texture
    
    def create_marble_texture(self, size):
        """Create marble texture"""
        texture = np.zeros((size, size), dtype=np.uint8)
        
        # Create swirl pattern
        for i in range(size):
            for j in range(size):
                angle = np.arctan2(i - size/2, j - size/2)
                radius = np.sqrt((i - size/2)**2 + (j - size/2)**2)
                value = int(128 + 50 * np.sin(radius * 0.1 + angle * 5))
                texture[i, j] = value
        
        return texture
    
    def get_brush(self, name, size, **kwargs):
        """Get brush by name"""
        if name in self.brushes:
            return self.brushes[name](size, **kwargs)
        return self.brushes["pencil"](size)
    
    def get_texture(self, name, size):
        """Get texture by name"""
        if name in self.textures:
            return self.textures[name](size)
        return self.textures["canvas"](size)