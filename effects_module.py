"""
Advanced Drawing Effects Module
Filters, textures, and special effects
"""

import cv2
import numpy as np

class EffectsEngine:
    """Advanced effects for AirCanvas Pro"""
    
    def __init__(self):
        self.effects = {
            "watercolor": self.watercolor_effect,
            "oil_paint": self.oil_paint_effect,
            "cartoon": self.cartoon_effect,
            "sketch": self.sketch_effect,
            "neon": self.neon_effect,
            "glow": self.glow_effect,
            "pixelate": self.pixelate_effect,
            "mosaic": self.mosaic_effect
        }
    
    def watercolor_effect(self, image):
        """Apply watercolor painting effect"""
        # Apply bilateral filter for smoothness
        smooth = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Reduce color palette
        quantized = self.quantize_colors(smooth, 12)
        
        # Add paper texture
        texture = self.add_paper_texture(quantized)
        
        return texture
    
    def oil_paint_effect(self, image, radius=5, levels=20):
        """Apply oil painting effect"""
        return cv2.xphoto.oilPainting(image, radius, levels)
    
    def cartoon_effect(self, image):
        """Apply cartoon effect"""
        # Edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, 
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 9, 9)
        
        # Color quantization
        color = self.quantize_colors(image, 8)
        
        # Combine edges with colors
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(color, edges)
        
        return cartoon
    
    def sketch_effect(self, image):
        """Apply pencil sketch effect"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        blur = cv2.GaussianBlur(inverted, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    def neon_effect(self, image):
        """Apply neon glow effect"""
        # Edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Dilate edges for neon effect
        kernel = np.ones((3,3), np.uint8)
        neon = cv2.dilate(edges, kernel, iterations=1)
        
        # Colorize
        neon_color = cv2.cvtColor(neon, cv2.COLOR_GRAY2BGR)
        neon_color[:,:,0] = 0  # Remove blue
        neon_color[:,:,1] = neon  # Green channel
        neon_color[:,:,2] = neon  # Red channel
        
        return neon_color
    
    def glow_effect(self, image):
        """Apply soft glow effect"""
        # Create blurred copy
        blur = cv2.GaussianBlur(image, (31, 31), 0)
        
        # Blend original with blur
        glow = cv2.addWeighted(image, 0.7, blur, 0.3, 0)
        
        return glow
    
    def pixelate_effect(self, image, pixel_size=10):
        """Apply pixelation effect"""
        h, w, _ = image.shape
        
        # Resize down
        small = cv2.resize(image, (w // pixel_size, h // pixel_size))
        
        # Resize up
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return pixelated
    
    def mosaic_effect(self, image, tile_size=20):
        """Apply mosaic tile effect"""
        h, w, _ = image.shape
        
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                # Get average color of tile
                tile = image[y:min(y+tile_size, h), x:min(x+tile_size, w)]
                avg_color = np.mean(tile, axis=(0,1)).astype(np.uint8)
                
                # Fill tile with average color
                image[y:min(y+tile_size, h), x:min(x+tile_size, w)] = avg_color
        
        return image
    
    def quantize_colors(self, image, k=8):
        """Reduce number of colors using K-means"""
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        quantized = quantized.reshape(image.shape)
        
        return quantized
    
    def add_paper_texture(self, image):
        """Add paper texture overlay"""
        # Create texture
        texture = np.random.randint(200, 255, image.shape, dtype=np.uint8)
        
        # Blend with image
        textured = cv2.addWeighted(image, 0.85, texture, 0.15, 0)
        
        return textured
    
    def apply_filter_chain(self, image, filters):
        """Apply multiple filters in sequence"""
        result = image.copy()
        
        for filter_name in filters:
            if filter_name in self.effects:
                result = self.effects[filter_name](result)
        
        return result