import numpy as np

class LBMSolver:
    def __init__(self, nx=400, ny=150, tau=0.55):
        self.nx = nx
        self.ny = ny
        self.tau = tau
        
        # D2Q9 lattice parameters
        self.NL = 9
        self.cxs = np.array([0, 1, 0, -1,  0, 1, -1, -1,  1])
        self.cys = np.array([0, 0, 1,  0, -1, 1,  1, -1, -1])
        self.weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        self.opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        
        # Macroscopic inlet velocity
        self.v_in = 0.1
        
        self.reset()
        
    def reset(self):
        # Initialize fields
        self.rho = np.ones((self.ny, self.nx))
        self.ux = np.ones((self.ny, self.nx)) * self.v_in
        self.uy = np.zeros((self.ny, self.nx))
        
        self.F = self.equilibrium(self.rho, self.ux, self.uy)
        self.obstacle = np.zeros((self.ny, self.nx), dtype=bool)
        
        self.lift = 0.0
        self.drag = 0.0

    def set_obstacle(self, obstacle_mask):
        """Update the boolean mask of solid boundaries"""
        self.obstacle = obstacle_mask
        self.ux[self.obstacle] = 0
        self.uy[self.obstacle] = 0
        # Re-calc equilibrium on boundaries to prevent instablities
        self.F[self.obstacle, :] = self.equilibrium(self.rho[self.obstacle], 
                                                    self.ux[self.obstacle], 
                                                    self.uy[self.obstacle])

    def equilibrium(self, rho, ux, uy):
        """Calculate the equilibrium distribution function"""
        # Support both full fields and masked 1D fields
        if rho.ndim == 2:
            Feq = np.zeros((self.ny, self.nx, self.NL))
            usqr = 1.5 * (ux**2 + uy**2)
            for i in range(self.NL):
                cu = 3 * (self.cxs[i]*ux + self.cys[i]*uy)
                Feq[:, :, i] = rho * self.weights[i] * (1 + cu + 0.5*cu**2 - usqr)
        else:
            Feq = np.zeros((rho.shape[0], self.NL))
            usqr = 1.5 * (ux**2 + uy**2)
            for i in range(self.NL):
                cu = 3 * (self.cxs[i]*ux + self.cys[i]*uy)
                Feq[:, i] = rho * self.weights[i] * (1 + cu + 0.5*cu**2 - usqr)
        return Feq
        
    def step(self):
        """Perform one LBM time step"""
        # 1. Collision step
        Feq = self.equilibrium(self.rho, self.ux, self.uy)
        F_post = self.F - (1.0 / self.tau) * (self.F - Feq)
        
        # 2. Obstacle Bounce-Back (No-slip boundary)
        # Particles hitting the obstacle bounce back in the opposite direction
        F_bnd = np.zeros_like(self.F)
        for i in range(self.NL):
             F_bnd[self.obstacle, i] = self.F[self.obstacle, self.opposite[i]]
             
        F_post[self.obstacle, :] = F_bnd[self.obstacle, :]
        
        # 3. Streaming step
        for i in range(self.NL):
            self.F[:, :, i] = np.roll(np.roll(F_post[:, :, i], self.cxs[i], axis=1), self.cys[i], axis=0)
            
        # 4. Boundary Conditions
        # Inlet (Left edge)
        self.ux[:, 0] = self.v_in
        self.uy[:, 0] = 0
        self.rho[:, 0] = 1.0
        F_inlet = self.equilibrium(self.rho[:, 0], self.ux[:, 0], self.uy[:, 0])
        for i in range(self.NL):
            self.F[:, 0, i] = F_inlet[:, i]
            
        # Outlet (Right edge) - Zero gradient
        for i in range(self.NL):
            self.F[:, -1, i] = self.F[:, -2, i]
            
        # Top and Bottom boundaries are currently periodic via np.roll
            
        # 5. Macroscopic variables update
        self.rho = np.sum(self.F, axis=2)
        
        # Prevent division by zero
        rho_safe = np.where(self.rho > 0.1, self.rho, 1.0)
        
        ux_new = sum(self.F[:,:,i] * self.cxs[i] for i in range(self.NL)) / rho_safe
        uy_new = sum(self.F[:,:,i] * self.cys[i] for i in range(self.NL)) / rho_safe
        
        self.ux = np.clip(ux_new, -0.5, 0.5) # Stability enforcing
        self.uy = np.clip(uy_new, -0.5, 0.5)
        
        self.ux[self.obstacle] = 0
        self.uy[self.obstacle] = 0
        
        # Calculate forces
        self.calculate_forces()

    def calculate_forces(self):
        """Calculate lift and drag via momentum exchange"""
        drag = 0.0
        lift = 0.0
        
        for i in range(1, self.NL):
             momentum_x = self.cxs[i]
             momentum_y = self.cys[i]
             # Momentum transferred to obstacle when bouncing back
             bounce_amount = self.F[self.obstacle, self.opposite[i]]
             drag += np.sum(2 * bounce_amount * momentum_x)
             # Note: positive Y in Pygame is down, but in physics up is positive.
             # We just calculate raw forces, can invert if needed on display.
             lift += np.sum(2 * bounce_amount * momentum_y)
             
        # Smooth forces for readable output
        smooth_factor = 0.05
        self.drag = self.drag * (1 - smooth_factor) + drag * smooth_factor
        self.lift = self.lift * (1 - smooth_factor) + lift * smooth_factor

    def get_velocity_field(self):
        return np.sqrt(self.ux**2 + self.uy**2)
        
    def get_pressure_field(self):
        # Pressure is proportional to density in LBM: P = rho * (cs^2)
        return self.rho
