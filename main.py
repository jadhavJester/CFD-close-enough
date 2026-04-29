import sys
import numpy as np
try:
    import pygame
except ImportError:
    print("Please install pygame via 'pip install pygame'")
    sys.exit(1)

from lbm_solver import LBMSolver
from visualizations import field_to_surface, add_obstacle_to_surface
from airfoils import generate_naca4, generate_kfm

def run():
    pygame.init()
    
    nx, ny = 300, 120
    scale = 4
    
    width, height = nx * scale, ny * scale
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Virtual Wind Tunnel - Aircraft Efficiency Simulation")
    
    font = pygame.font.SysFont(None, 24)
    big_font = pygame.font.SysFont(None, 34)
    
    print("Initializing Simulation Engine...")
    solver = LBMSolver(nx=nx, ny=ny, tau=0.55)
    
    obstacle = np.zeros((ny, nx), dtype=bool)
    
    # State variables
    current_airfoil = 'NACA2412'
    aoa = 5.0
    wind_speed = 20.0
    aircraft_speed = 100.0
    
    input_rect = pygame.Rect(width - 160, 10, 140, 32)
    input_active = False
    user_text = ''
    
    def parse_naca(naca_str):
        if naca_str.startswith('NACA') and len(naca_str) == 8:
            return int(naca_str[4]), int(naca_str[5]), int(naca_str[6:8])
        return None

    def build_naca(m, p, t):
        return f"NACA{m}{p}{t:02d}"

    def apply_airfoil():
        if current_airfoil.startswith('NACA'):
            naca_str = current_airfoil[4:]
            mask = generate_naca4((ny, nx), nx//4, ny//2, 80, naca_str, aoa)
        else:
            try:
                kfm_type = int(current_airfoil.replace('KFM', ''))
            except:
                kfm_type = 2
            mask = generate_kfm((ny, nx), nx//4, ny//2, 80, kfm_type, 0.09, aoa)
            
        obstacle[:] = mask
        solver.set_obstacle(obstacle)
        solver.reset()
        solver.set_obstacle(obstacle)
        
        # update inlet speed based on total requested speed (cap dimensionless speed for LBM stability)
        total_speed = wind_speed + aircraft_speed
        safe_v_in = 0.15 * (total_speed / 150.0)
        solver.v_in = max(0.01, min(safe_v_in, 0.25))

    apply_airfoil()
    
    clock = pygame.time.Clock()
    running = True
    
    sim_steps_per_frame = 10 
    drawing = False
    erasing = False
    brush_size = 3
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if input_rect.collidepoint(event.pos):
                        input_active = True
                    else:
                        input_active = False
                        drawing = True
                elif event.button == 3:
                    erasing = True
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drawing = False
                elif event.button == 3:
                    erasing = False
                    
            elif event.type == pygame.MOUSEMOTION:
                if drawing or erasing:
                    mx, my = pygame.mouse.get_pos()
                    gx = int(mx / scale)
                    gy = int(my / scale)
                    
                    if 0 <= gx < nx and 0 <= gy < ny:
                        for i in range(max(0, gy - brush_size), min(ny, gy + brush_size)):
                            for j in range(max(0, gx - brush_size), min(nx, gx + brush_size)):
                                if (i - gy)**2 + (j - gx)**2 <= brush_size**2:
                                    obstacle[i, j] = drawing
                        solver.set_obstacle(obstacle)
                        
            elif event.type == pygame.KEYDOWN:
                if input_active:
                    if event.key == pygame.K_RETURN:
                        if len(user_text) == 4 and user_text.isdigit():
                            current_airfoil = f'NACA{user_text}'
                            apply_airfoil()
                        input_active = False
                        user_text = ''
                    elif event.key == pygame.K_ESCAPE:
                        input_active = False
                        user_text = ''
                    elif event.key == pygame.K_BACKSPACE:
                        user_text = user_text[:-1]
                    elif event.unicode.isdigit() and len(user_text) < 4:
                        user_text += event.unicode
                else:
                    if event.key == pygame.K_r:
                        solver.reset()
                        solver.set_obstacle(obstacle)
                    elif event.key == pygame.K_c:
                        obstacle.fill(False)
                        solver.set_obstacle(obstacle)
                    elif event.key == pygame.K_1:
                        current_airfoil = 'NACA0012'
                        apply_airfoil()
                    elif event.key == pygame.K_2:
                        current_airfoil = 'NACA2412'
                        apply_airfoil()
                    elif event.key == pygame.K_3:
                        current_airfoil = 'NACA4415'
                        apply_airfoil()
                    elif event.key == pygame.K_UP: # AOA increment
                        aoa += 1.0
                        apply_airfoil()
                    elif event.key == pygame.K_DOWN:
                        aoa -= 1.0
                        apply_airfoil()
                    elif event.key == pygame.K_RIGHT: # Speed increment
                        aircraft_speed += 10.0
                        apply_airfoil()
                    elif event.key == pygame.K_LEFT:
                        aircraft_speed = max(0.0, aircraft_speed - 10.0)
                        apply_airfoil()
                    elif event.key == pygame.K_q: # Increase camber
                        params = parse_naca(current_airfoil)
                        if params:
                            current_airfoil = build_naca(min(params[0] + 1, 9), params[1], params[2])
                            apply_airfoil()
                    elif event.key == pygame.K_a: # Decrease camber
                        params = parse_naca(current_airfoil)
                        if params:
                            current_airfoil = build_naca(max(params[0] - 1, 0), params[1], params[2])
                            apply_airfoil()
                    elif event.key == pygame.K_w: # Increase camber pos
                        params = parse_naca(current_airfoil)
                        if params:
                            current_airfoil = build_naca(params[0], min(params[1] + 1, 9), params[2])
                            apply_airfoil()
                    elif event.key == pygame.K_s: # Decrease camber pos
                        params = parse_naca(current_airfoil)
                        if params:
                            current_airfoil = build_naca(params[0], max(params[1] - 0, 0), params[2])
                            apply_airfoil()
                    elif event.key == pygame.K_e: # Increase thickness
                        params = parse_naca(current_airfoil)
                        if params:
                            current_airfoil = build_naca(params[0], params[1], min(params[2] + 1, 40))
                            apply_airfoil()
                    elif event.key == pygame.K_d: # Decrease thickness
                        params = parse_naca(current_airfoil)
                        if params:
                            current_airfoil = build_naca(params[0], params[1], max(params[2] - 1, 1))
                            apply_airfoil()

        # Physics computations
        for _ in range(sim_steps_per_frame):
            solver.step()

        # Rendering
        vel_field = solver.get_velocity_field()
        
        surf = field_to_surface(vel_field, vmin=0.0, vmax=solver.v_in * 2.0, cmap='jet')
        add_obstacle_to_surface(surf, solver.obstacle, color=(20, 20, 20))
        
        if surf is not None:
            scaled_surf = pygame.transform.scale(surf, (width, height))
            screen.blit(scaled_surf, (0, 0))
            
        # Draw HUD Metrics
        # Map LBM to physical parameters inspired by virtual-wind-tunnel
        total_speed = wind_speed + aircraft_speed
        velocity_scale = total_speed / solver.v_in if solver.v_in > 0 else 1.0
        
        max_vel = np.max(vel_field) * velocity_scale
        rho = 1.225
        p_dyn = 0.5 * rho * abs((max_vel**2) - (total_speed**2))
        
        force_scale = (total_speed**2) * rho * 5.0
        lift = solver.lift * force_scale
        drag = solver.drag * force_scale
        
        weight = abs(lift)
        thrust = abs(drag)
        efficiency = abs(lift) / abs(drag) if abs(drag) > 1e-4 else 0.0
        
        hud_bg = pygame.Surface((340, 240))
        hud_bg.set_alpha(180)
        hud_bg.fill((0, 0, 0))
        screen.blit(hud_bg, (10, 10))
        
        # Draw Output
        screen.blit(font.render(f"Airfoil: {current_airfoil} | AoA: {aoa:.1f}°", True, (255, 200, 100)), (20, 20))
        screen.blit(font.render(f"Speed (Wind+Craft): {total_speed:.1f} m/s", True, (255, 200, 200)), (20, 45))
        
        screen.blit(font.render(f"Max Velocity: {max_vel:.2f} m/s", True, (255, 85, 85)), (20, 75))
        screen.blit(font.render(f"Dynamic Pressure: {p_dyn:.2f} Pa", True, (85, 170, 255)), (20, 100))
        
        screen.blit(font.render(f"Lift Force (L): {lift:.2f} N", True, (85, 255, 85)), (20, 130))
        screen.blit(font.render(f"Drag Force (D): {drag:.2f} N", True, (255, 170, 0)), (20, 155))
        
        screen.blit(font.render(f"Req. Weight (W=L): {weight:.2f} N", True, (200, 200, 200)), (20, 185))
        screen.blit(font.render(f"Req. Thrust (T=D): {thrust:.2f} N", True, (255, 120, 0)), (20, 210))
        
        # Calculate & Output efficiency 
        eff_color = (100, 255, 100) if efficiency > 1.5 else (200, 200, 200)
        if efficiency < 0.5: eff_color = (255, 100, 100)
        screen.blit(big_font.render(f"L/D Efficiency: {efficiency:.2f}", True, eff_color), (screen.get_width() - 250, screen.get_height() - 40))
        
        # Draw Input Box
        color = pygame.Color('lightskyblue3') if input_active else pygame.Color('gray15')
        pygame.draw.rect(screen, color, input_rect, 2)
        screen.blit(font.render(f"NACA: {user_text}", True, (255, 255, 255)), (input_rect.x + 5, input_rect.y + 8))
        
        # Draw instructions on the bottom
        text_inst1 = font.render("Draw: L-Click | Erase: R-Click | 'C': Clear | '1-3': Presets | L/R Arrows: Speed | U/D Arrows: AoA", True, (200, 200, 200))
        text_inst2 = font.render("NACA Mod: Q/A (Camber) | W/S (Pos) | E/D (Thickness)", True, (200, 200, 200))
        screen.blit(text_inst1, (10, height - 45))
        screen.blit(text_inst2, (10, height - 25))

        pygame.display.flip()
        dt = clock.tick(60) 
        pygame.display.set_caption(f"Virtual Wind Tunnel - Aircraft Efficiency | FPS: {clock.get_fps():.1f}")

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    run()
