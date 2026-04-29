import numpy as np

def generate_naca4(shape, cx, cy, chord, naca_string, aoa_deg=0):
    """
    Generates a boolean mask on a 2D grid containing a NACA 4-digit airfoil.
    """
    ny, nx = shape
    mask = np.zeros((ny, nx), dtype=bool)
    
    if len(naca_string) != 4:
        return mask
        
    m = int(naca_string[0]) / 100.0
    p = int(naca_string[1]) / 10.0
    t = int(naca_string[2:]) / 100.0
    
    aoa_rad = np.radians(aoa_deg)
    cos_a = np.cos(-aoa_rad)
    sin_a = np.sin(-aoa_rad)
    
    # We will iterate over a generous bounding box to map coords
    R = chord * 1.5
    y_min, y_max = max(0, int(cy - R)), min(ny, int(cy + R))
    x_min, x_max = max(0, int(cx - R)), min(nx, int(cx + R))
    
    if y_max <= y_min or x_max <= x_min:
        return mask
        
    y_idx, x_idx = np.ogrid[y_min:y_max, x_min:x_max]
    
    dx = x_idx - cx
    dy = y_idx - cy
    
    # Inverse rotation to align with standard normalized coordinate space
    x_rot = dx * cos_a - dy * sin_a
    y_rot = dx * sin_a + dy * cos_a
    
    # Shift from center to leading-edge
    x_local = x_rot + chord / 2.0
    y_local = -y_rot # Invert Y for math standard upward positive
    
    x_norm = x_local / chord
    valid_mask = (x_norm >= 0.0) & (x_norm <= 1.0)
    
    # Evaluate thickness and camber only where valid
    x_n_val = x_norm[valid_mask]
    
    yt = 5.0 * t * (0.2969 * np.sqrt(x_n_val) - 0.1260 * x_n_val - 
                    0.3516 * x_n_val**2 + 0.2843 * x_n_val**3 - 0.1015 * x_n_val**4)
                    
    yc = np.zeros_like(x_n_val)
    if p > 0:
        front = x_n_val < p
        back = slice(None) if p == 0 else ~front # safe check
        
        yc[front] = (m / p**2) * (2 * p * x_n_val[front] - x_n_val[front]**2)
        yc[back] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x_n_val[back] - x_n_val[back]**2)
        
    y_upper = (yc + yt) * chord
    y_lower = (yc - yt) * chord
    
    y_loc_val = y_local[valid_mask]
    inside = (y_loc_val <= y_upper) & (y_loc_val >= y_lower)
    
    sub_mask = np.zeros_like(valid_mask, dtype=bool)
    sub_mask[valid_mask] = inside
    
    mask[y_min:y_max, x_min:x_max] = sub_mask
    return mask

def generate_kfm(shape, cx, cy, chord, kfm_type, thickness_pct=0.09, aoa_deg=0):
    """
    Generates a boolean mask for stepped Kline-Fogleman airfoils.
    """
    ny, nx = shape
    mask = np.zeros((ny, nx), dtype=bool)
    
    aoa_rad = np.radians(aoa_deg)
    cos_a = np.cos(-aoa_rad)
    sin_a = np.sin(-aoa_rad)
    
    R = chord * 1.5
    y_min, y_max = max(0, int(cy - R)), min(ny, int(cy + R))
    x_min, x_max = max(0, int(cx - R)), min(nx, int(cx + R))
    
    if y_max <= y_min or x_max <= x_min:
        return mask
        
    y_idx, x_idx = np.ogrid[y_min:y_max, x_min:x_max]
    
    dx = x_idx - cx
    dy = y_idx - cy
    
    x_rot = dx * cos_a - dy * sin_a
    y_rot = dx * sin_a + dy * cos_a
    
    x_local = x_rot + chord / 2.0
    y_local = -y_rot
    
    x_norm = x_local / chord
    valid_mask = (x_norm >= 0.0) & (x_norm <= 1.0)
    
    x_n_val = x_norm[valid_mask]
    y_loc_val = y_local[valid_mask] / chord
    
    # Base thin flat surface line
    base_t = thickness_pct * 0.4
    y_upper = np.ones_like(x_n_val) * base_t
    y_lower = np.ones_like(x_n_val) * (-base_t)
    
    # Leading edge simple rounding approximation
    le = np.where(x_n_val < 0.05, np.sqrt(x_n_val / 0.05), 1.0)
    y_upper *= le
    y_lower *= le
    
    thickness = thickness_pct
    
    # Apply steps based on kfm type
    if kfm_type == 1:
        # Bottom step at 40%
        y_lower = np.where(x_n_val < 0.40, -thickness, y_lower)
    elif kfm_type == 2:
        # Top step at 50%
        y_upper = np.where(x_n_val < 0.50, thickness, y_upper)
    elif kfm_type == 3:
        # Two top steps at 50%, 75%
        step1 = np.where(x_n_val < 0.50, thickness, y_upper)
        y_upper = np.where(x_n_val < 0.75, thickness * 0.70, step1)
    elif kfm_type == 4:
        # Sym steps at 50%
        y_upper = np.where(x_n_val < 0.50, thickness, y_upper)
        y_lower = np.where(x_n_val < 0.50, -thickness, y_lower)
        
    inside = (y_loc_val <= y_upper) & (y_loc_val >= y_lower)
    
    sub_mask = np.zeros_like(valid_mask, dtype=bool)
    sub_mask[valid_mask] = inside
    
    mask[y_min:y_max, x_min:x_max] = sub_mask
    return mask
