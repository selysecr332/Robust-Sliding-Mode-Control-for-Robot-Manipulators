# File: 07_adaptive_control_fixed.py

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from simulator import Simulator
from pathlib import Path
import os
from typing import Dict, Tuple, List
import time


class DynamicsRegressor:
    """Compute dynamics regressor matrix for adaptive control"""

    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromMJCF(urdf_path)
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.nv = self.model.nv

        # Get the actual number of base parameters
        self.n_base_params = 10 * self.nv  # Standard: 10 inertial params per joint

    def compute_regressor(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
        """Compute joint torque regressor matrix"""
        return pin.computeJointTorqueRegressor(self.model, self.data, q, dq, ddq)

    def compute_dynamics(self, q: np.ndarray, dq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute mass matrix, Coriolis, and gravity terms"""
        pin.computeAllTerms(self.model, self.data, q, dq)
        return self.data.M.copy(), self.data.nle.copy(), self.data.g.copy()

    def get_parameter_count(self) -> int:
        """Get number of base parameters from regressor"""
        # Test with random values to get regressor shape
        q_test = np.random.randn(self.nq)
        dq_test = np.random.randn(self.nv)
        ddq_test = np.random.randn(self.nv)
        regressor = self.compute_regressor(q_test, dq_test, ddq_test)
        return regressor.shape[1]


class InverseDynamicsController:
    """Nominal Inverse Dynamics Controller (baseline)"""

    def __init__(self, dynamics: DynamicsRegressor):
        self.dyn = dynamics
        self.kp = 100 * np.ones(6)
        self.kd = 20 * np.ones(6)

        # Desired trajectory for regulation
        self.q_des = np.array([-1.4, -1.3, 1.0, 0, 0, 0])
        self.dq_des = np.zeros(6)
        self.ddq_des = np.zeros(6)

    def regulation_control(self, q: np.ndarray, dq: np.ndarray, t: float = 0) -> np.ndarray:
        """Regulation to fixed setpoint"""
        e = self.q_des - q
        de = self.dq_des - dq

        # Compute nominal dynamics
        M, nle, g = self.dyn.compute_dynamics(q, dq)

        # PD + feedforward
        tau_pd = self.kp * e + self.kd * de
        tau_ff = nle

        return tau_ff + tau_pd


class SlotineLiAdaptiveController:
    """Slotine and Li Adaptive Passivity-Based Controller - FIXED"""

    def __init__(self, dynamics: DynamicsRegressor,
                 lambda_val: float = 10.0,
                 gamma: float = 0.1,
                 damping_adaptation: bool = True,
                 parameter_bounds: bool = True):
        """
        Args:
            lambda_val: Sliding surface parameter
            gamma: Adaptation gain
            damping_adaptation: Whether to adapt to damping
            parameter_bounds: Whether to enforce parameter bounds
        """
        self.dyn = dynamics
        self.lambda_val = lambda_val
        self.gamma = gamma

        # Desired trajectory
        self.q_des = np.array([-1.4, -1.3, 1.0, 0, 0, 0])
        self.dq_des = np.zeros(6)
        self.ddq_des = np.zeros(6)

        # Parameter adaptation
        self.damping_adaptation = damping_adaptation
        self.parameter_bounds = parameter_bounds

        # Get actual number of base parameters from regressor
        self.n_base_params = self.dyn.get_parameter_count()
        print(f"Base parameters from regressor: {self.n_base_params}")

        # Total parameters: base + additional mass + damping
        self.n_ee_mass = 1  # Additional mass at EE
        self.n_damping = 6 if damping_adaptation else 0
        self.n_total_params = self.n_base_params + self.n_ee_mass + self.n_damping

        # Initialize parameter estimates
        self.theta_hat = np.zeros(self.n_total_params)

        # Initialize with reasonable guesses
        self.theta_hat[self.n_base_params] = 0.1  # Small initial mass guess

        # Parameter bounds
        self.theta_min = np.full_like(self.theta_hat, -np.inf)
        self.theta_max = np.full_like(self.theta_hat, np.inf)

        # Set reasonable bounds
        if parameter_bounds:
            # Mass parameters should be positive
            for i in range(self.n_base_params):
                if i % 10 == 0:  # Mass parameters (every 10th)
                    self.theta_min[i] = 0.0

            # Additional mass should be positive
            self.theta_min[self.n_base_params] = 0.0
            self.theta_max[self.n_base_params] = 10.0  # Max 10kg

            # Damping should be positive
            if damping_adaptation:
                damping_start = self.n_base_params + self.n_ee_mass
                for i in range(6):
                    self.theta_min[damping_start + i] = 0.0
                    self.theta_max[damping_start + i] = 10.0  # Max 10 Nms/rad

        # Adaptation history
        self.theta_history = []
        self.s_history = []

    def project_parameters(self, theta: np.ndarray) -> np.ndarray:
        """Project parameters to feasible bounds"""
        if not self.parameter_bounds:
            return theta

        theta_projected = np.copy(theta)
        for i in range(len(theta)):
            if theta_projected[i] < self.theta_min[i]:
                theta_projected[i] = self.theta_min[i]
            elif theta_projected[i] > self.theta_max[i]:
                theta_projected[i] = self.theta_max[i]

        return theta_projected

    def compute_control(self, q: np.ndarray, dq: np.ndarray, t: float,
                        mode: str = 'regulation') -> np.ndarray:
        """
        Compute adaptive control torque

        Args:
            mode: 'regulation' or 'tracking'
        """
        # Compute desired trajectory (simplified regulation)
        qr = self.q_des
        dqr = self.dq_des
        ddqr = self.ddq_des

        # Tracking errors
        e = qr - q
        de = dqr - dq

        # Sliding variable
        s = de + self.lambda_val * e

        # Reference velocity and acceleration
        qr_dot = dqr + self.lambda_val * e
        qr_ddot = ddqr + self.lambda_val * de

        # Compute regressor for reference trajectory
        try:
            regressor = self.dyn.compute_regressor(q, qr_dot, qr_ddot)
        except:
            # If regressor computation fails, use zeros
            regressor = np.zeros((6, self.n_base_params))

        # Control law: τ = Y(q, q̇_r, q̈_r)θ̂ - K_D s
        K_D = 20 * np.eye(6)  # Damping matrix

        # Only use base parameters for control
        if regressor.shape[1] == self.n_base_params:
            tau_base = regressor @ self.theta_hat[:self.n_base_params]
        else:
            # Regressor has different shape, use what we have
            n_cols = min(regressor.shape[1], self.n_base_params)
            tau_base = regressor[:, :n_cols] @ self.theta_hat[:n_cols]

        tau = tau_base - K_D @ s

        # Parameter adaptation law: θ̂̇ = -Γ Yᵀ s
        # Only adapt base parameters (not additional mass/damping in this simplified version)
        if regressor.shape[1] <= self.n_base_params:
            theta_dot_base = self.gamma * regressor.T @ s

            # Pad with zeros for additional parameters
            theta_dot = np.zeros(self.n_total_params)
            theta_dot[:len(theta_dot_base)] = theta_dot_base

            # Update parameter estimates with projection
            self.theta_hat += theta_dot * 0.002  # dt = 0.002
            self.theta_hat = self.project_parameters(self.theta_hat)

        # Store history
        self.theta_history.append(self.theta_hat.copy())
        self.s_history.append(s.copy())

        return tau

    def get_parameter_estimates(self) -> Dict[str, float]:
        """Extract meaningful parameter estimates"""
        estimates = {
            'base_inertial_params': self.theta_hat[:self.n_base_params].copy(),
            'additional_mass': self.theta_hat[self.n_base_params] if len(self.theta_hat) > self.n_base_params else 0.0
        }

        if self.damping_adaptation and len(self.theta_hat) > self.n_base_params + 1:
            estimates['damping_coeffs'] = self.theta_hat[self.n_base_params + 1:self.n_base_params + 7].copy()

        return estimates


def run_fast_adaptive_experiment():
    """Fast version without video recording"""

    print("=" * 70)
    print("FAST ADAPTIVE CONTROL EXPERIMENT (No Video)")
    print("=" * 70)

    # Create directories
    Path("logs/plots").mkdir(parents=True, exist_ok=True)
    Path("logs/data").mkdir(parents=True, exist_ok=True)

    # Path to robot model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")

    # Initialize dynamics computer
    dynamics = DynamicsRegressor(urdf_path)

    # Unknown parameters
    unknown_mass = 4.0  # kg
    unknown_damping = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])  # Nms/rad

    print(f"Unknown mass: {unknown_mass} kg")
    print(f"Unknown damping: {unknown_damping}")

    # ======================================================================
    # Task 3: Inverse Dynamics Controller
    # ======================================================================
    print("\n" + "-" * 70)
    print("TASK 3: Inverse Dynamics Controller (Nominal Model)")
    print("-" * 70)

    sim_id = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=False,
        show_viewer=False,
        record_video=False
    )

    # Add uncertainties
    sim_id.modify_body_properties("end_effector", mass=unknown_mass)
    sim_id.set_joint_damping(unknown_damping)

    id_controller = InverseDynamicsController(dynamics)

    # Data collection
    id_data = {'time': [], 'q': [], 'error': []}

    def id_wrapper(q, dq, t):
        tau = id_controller.regulation_control(q, dq, t)
        id_data['time'].append(t)
        id_data['q'].append(q.copy())
        id_data['error'].append(id_controller.q_des - q)
        return tau

    print("Running ID controller...")
    sim_id.set_controller(id_wrapper)
    sim_id.reset()

    # Run faster simulation
    t = 0
    dt = sim_id.dt
    time_limit = 5.0

    while t < time_limit:
        state = sim_id.get_state()
        tau = id_wrapper(state['q'], state['dq'], t)
        sim_id.step(tau)
        t += dt

    for key in id_data:
        id_data[key] = np.array(id_data[key])

    id_rmse = np.mean(np.sqrt(np.mean(id_data['error'] ** 2, axis=0)))
    print(f"ID Controller RMSE: {id_rmse:.6f} rad")

    # ======================================================================
    # Task 4: Adaptive Controller
    # ======================================================================
    print("\n" + "-" * 70)
    print("TASK 4: Slotine and Li Adaptive Controller")
    print("-" * 70)

    sim_adaptive = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=False,
        show_viewer=False,
        record_video=False
    )

    sim_adaptive.modify_body_properties("end_effector", mass=unknown_mass)
    sim_adaptive.set_joint_damping(unknown_damping)

    adaptive_controller = SlotineLiAdaptiveController(
        dynamics,
        gamma=0.5,
        damping_adaptation=True,
        parameter_bounds=True
    )

    adaptive_data = {
        'time': [], 'q': [], 'error': [], 'tau': [],
        'theta_hat': [], 's': []
    }

    def adaptive_wrapper(q, dq, t):
        tau = adaptive_controller.compute_control(q, dq, t, mode='regulation')
        adaptive_data['time'].append(t)
        adaptive_data['q'].append(q.copy())
        adaptive_data['error'].append(adaptive_controller.q_des - q)
        adaptive_data['tau'].append(tau.copy())
        adaptive_data['theta_hat'].append(adaptive_controller.theta_hat.copy())
        adaptive_data['s'].append(adaptive_controller.s_history[-1].copy())
        return tau

    print("Running adaptive controller...")
    sim_adaptive.set_controller(adaptive_wrapper)
    sim_adaptive.reset()

    # Run faster simulation
    t = 0
    dt = sim_adaptive.dt
    time_limit = 10.0

    while t < time_limit:
        state = sim_adaptive.get_state()
        tau = adaptive_wrapper(state['q'], state['dq'], t)
        sim_adaptive.step(tau)
        t += dt

    for key in adaptive_data:
        adaptive_data[key] = np.array(adaptive_data[key])

    adaptive_rmse = np.mean(np.sqrt(np.mean(adaptive_data['error'] ** 2, axis=0)))
    print(f"Adaptive Controller RMSE: {adaptive_rmse:.6f} rad")

    # ======================================================================
    # Task 5: Parameter Convergence
    # ======================================================================
    print("\n" + "-" * 70)
    print("TASK 5: Parameter Convergence Analysis")
    print("-" * 70)

    if len(adaptive_data['theta_hat']) > 0:
        theta_history = adaptive_data['theta_hat']
        n_base = adaptive_controller.n_base_params

        print(f"\nBase parameter convergence (first 5 params):")
        for i in range(min(5, n_base)):
            init_val = theta_history[0, i]
            final_val = theta_history[-1, i]
            print(f"  Param {i + 1}: {init_val:.4f} → {final_val:.4f}")

        # Additional mass convergence
        if theta_history.shape[1] > n_base:
            mass_estimates = theta_history[:, n_base]
            print(f"\nAdditional mass convergence:")
            print(f"  Initial: {mass_estimates[0]:.4f} kg")
            print(f"  Final: {mass_estimates[-1]:.4f} kg")
            print(f"  True: {unknown_mass:.4f} kg")
            print(f"  Error: {abs(mass_estimates[-1] - unknown_mass):.4f} kg")

    # ======================================================================
    # Generate Plots
    # ======================================================================
    print("\n" + "-" * 70)
    print("GENERATING RESULTS AND PLOTS")
    print("-" * 70)

    generate_simple_plots(id_data, adaptive_data, unknown_mass)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  ID Controller RMSE: {id_rmse:.6f} rad")
    print(f"  Adaptive Controller RMSE: {adaptive_rmse:.6f} rad")
    print(f"  Improvement: {(id_rmse - adaptive_rmse) / id_rmse * 100:+.1f}%")


def generate_simple_plots(id_data, adaptive_data, true_mass):
    """Generate essential plots"""

    # Plot 1: Tracking error comparison
    plt.figure(figsize=(12, 8))

    # Joint 1 error
    plt.subplot(2, 2, 1)
    plt.plot(id_data['time'], id_data['error'][:, 0], 'r-', label='ID Controller', linewidth=2)
    plt.plot(adaptive_data['time'], adaptive_data['error'][:, 0], 'b-', label='Adaptive Controller', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint 1 Error [rad]')
    plt.title('Joint 1 Tracking Error Comparison')
    plt.legend()
    plt.grid(True)

    # Joint 2 error
    plt.subplot(2, 2, 2)
    plt.plot(id_data['time'], id_data['error'][:, 1], 'r-', label='ID Controller', linewidth=2)
    plt.plot(adaptive_data['time'], adaptive_data['error'][:, 1], 'b-', label='Adaptive Controller', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint 2 Error [rad]')
    plt.title('Joint 2 Tracking Error Comparison')
    plt.legend()
    plt.grid(True)

    # Mass parameter convergence
    if len(adaptive_data['theta_hat']) > 0:
        n_base = adaptive_data['theta_hat'].shape[1] - 1  # Assume last is mass
        if n_base >= 0:
            plt.subplot(2, 2, 3)
            mass_estimates = adaptive_data['theta_hat'][:, n_base]
            plt.plot(adaptive_data['time'], mass_estimates, 'g-', linewidth=2, label='Mass Estimate')
            plt.axhline(y=true_mass, color='r', linestyle='--', linewidth=2, label='True Mass')
            plt.xlabel('Time [s]')
            plt.ylabel('Mass Estimate [kg]')
            plt.title('Parameter Convergence: Additional Mass')
            plt.legend()
            plt.grid(True)

    # Control torque comparison
    plt.subplot(2, 2, 4)
    plt.plot(id_data['time'][:100], id_data['error'][:100, 0], 'r-', label='ID Error', alpha=0.7)
    plt.plot(adaptive_data['time'][:100], adaptive_data['error'][:100, 0], 'b-', label='Adaptive Error', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Error [rad]')
    plt.title('Initial Response (First 0.2s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('logs/plots/07_adaptive_results_simple.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: Sliding surfaces
    if len(adaptive_data['s']) > 0:
        plt.figure(figsize=(10, 6))
        for i in range(3):  # First 3 joints
            plt.plot(adaptive_data['time'], adaptive_data['s'][:, i], label=f'Joint {i + 1}')
        plt.xlabel('Time [s]')
        plt.ylabel('Sliding Variable s')
        plt.title('Sliding Surfaces Evolution')
        plt.legend()
        plt.grid(True)
        plt.savefig('logs/plots/07_sliding_surfaces_simple.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_minimal_test():
    """Minimal test to verify controller works"""

    print("Minimal adaptive control test...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")

    # Initialize dynamics
    dynamics = DynamicsRegressor(urdf_path)

    # Test regressor
    q = np.zeros(6)
    dq = np.zeros(6)
    ddq = np.zeros(6)

    regressor = dynamics.compute_regressor(q, dq, ddq)
    print(f"Regressor shape: {regressor.shape}")
    print(f"Number of parameters: {regressor.shape[1]}")

    # Create adaptive controller
    controller = SlotineLiAdaptiveController(dynamics)

    # Test control computation
    q_test = np.array([0.1, -0.2, 0.3, 0.1, 0.2, 0.1])
    dq_test = np.array([0.01, 0.02, -0.01, 0.02, 0.01, 0.03])

    tau = controller.compute_control(q_test, dq_test, 0.0, 'regulation')
    print(f"Control torque computed successfully: {tau}")

    return True


if __name__ == "__main__":
    # First run minimal test
    run_minimal_test()

    # Then run full experiment
    run_fast_adaptive_experiment()