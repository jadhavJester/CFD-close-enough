import numpy as np
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


def field_to_surface(field, vmin=None, vmax=None, cmap='jet'):
    """
    Converts a 2D numpy array field into a PyGame surface with a heatmap colormap.
    Optimized for real-time performance using basic numpy operations.
    """
    if not PYGAME_AVAILABLE:
        return None

    if vmin is None:
        vmin = field.min()
    if vmax is None:
        vmax = field.max()
        
    range_val = vmax - vmin
    if range_val == 0:
        range_val = 1.0
        
    # Normalize between 0 and 1
    norm = np.clip((field - vmin) / range_val, 0, 1)
    
    # Create simple colormap (similar to 'jet': Blue -> Cyan -> Green -> Yellow -> Red)
    if cmap == 'jet':
        # r: 0 until 0.35, then linear to 1 at 0.65, then 1
        r = np.clip((norm - 0.35) / 0.3, 0, 1)
        # g: 0 until 0.1, then linear to 1 at 0.35, stays 1 until 0.65, down to 0 at 0.9
        g = np.clip((norm - 0.1) / 0.25, 0, 1) - np.clip((norm - 0.65) / 0.25, 0, 1)
        # b: 1 until 0.35, down to 0 at 0.65
        b = 1.0 - np.clip((norm - 0.35) / 0.3, 0, 1)
        b = np.clip(b + np.clip((0.1 - norm)/0.1, 0, 1), 0, 1) # Add bit of blue at bottom
    else:
        # Grayscale fallback
        r = g = b = norm
        
    r = (r * 255).astype(np.uint8)
    g = (g * 255).astype(np.uint8)
    b = (b * 255).astype(np.uint8)
    
    # field is (ny, nx). RGB stack is (ny, nx, 3)
    rgb = np.stack((r, g, b), axis=-1)
    
    # pygame surfarray expects shape (width, height, 3) so transpose axes 0 and 1
    rgb = np.transpose(rgb, (1, 0, 2))
    
    return pygame.surfarray.make_surface(rgb)

def add_obstacle_to_surface(surface, obstacle_mask, color=(100, 100, 100)):
    """
    Overlays the obstacle onto the PyGame surface.
    obstacle_mask is (ny, nx), True where obstacle exists.
    """
    if surface is None:
        return
        
    # getting pixel data and setting the obstacle mask
    pixel_array = pygame.surfarray.pixels3d(surface)
    # Pygame array is (nx, ny, 3), obstacle is (ny, nx). We transpose mask.
    mask_T = obstacle_mask.T
    
    pixel_array[mask_T] = color
    # Let go of pixel_array to unlock the surface
    del pixel_array
    
    return surface
