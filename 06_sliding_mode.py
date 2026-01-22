# File: 06_sliding_mode.py
"""

Implementation of:
1. Inverse Dynamics Controller (40 points)
2. Sliding Mode Controller with uncertainties (40 points)
3. Boundary Layer Analysis (20 points)

"""

import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from simulator import Simulator
from pathlib import Path
import os


# ============================================================================
# PART 1: INVERSE DYNAMICS CONTROLLER (40 points)
# ============================================================================

class InverseDynamicsController:

    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromMJCF(urdf_path)
        self.data = self.model.createData()

        # PD gains as specified in assignment
        self.kp = 100.0 * np.ones(6)
        self.kd = 20.0 * np.ones(6)

        # Desired trajectory
        self.q_des = np.array([-1.4, -1.3, 1.0, 0, 0, 0])
        self.dq_des = np.zeros(6)

    def compute_torque(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        # PD feedback
        e = self.q_des - q
        de = self.dq_des - dq
        tau_pd = self.kp * e + self.kd * de

        # Feedforward dynamics compensation
        pin.computeAllTerms(self.model, self.data, q, dq)
        tau_ff = self.data.nle  # C*dq + g

        return tau_ff + tau_pd


# ============================================================================
# PART 2: SLIDING MODE CONTROLLER (40 points)
# ============================================================================

class SlidingModeController:

    def __init__(self, urdf_path: str, use_bl: bool = True, phi: float = 0.1):
        self.model = pin.buildModelFromMJCF(urdf_path)
        self.data = self.model.createData()

        # SMC parameters
        self.lambda_val = 10.0  # Sliding surface parameter
        self.K = 50.0  # Discontinuous gain
        self.phi = phi  # Boundary layer thickness
        self.use_boundary_layer = use_bl

        self.rho = 20.0  # Total uncertainty bound

        # Desired trajectory
        self.q_des = np.array([-1.4, -1.3, 1.0, 0, 0, 0])
        self.dq_des = np.zeros(6)
        self.ddq_des = np.zeros(6)

    def sat(self, s: np.ndarray) -> np.ndarray:
        if not self.use_boundary_layer:
            return np.sign(s)

        result = np.zeros_like(s)
        abs_s = np.abs(s)
        mask = abs_s <= self.phi
        result[mask] = s[mask] / self.phi
        result[~mask] = np.sign(s[~mask])
        return result

    def compute_torque(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        # Tracking errors
        e = self.q_des - q
        de = self.dq_des - dq

        # Sliding surface
        s = de + self.lambda_val * e

        # Compute nominal dynamics
        pin.computeAllTerms(self.model, self.data, q, dq)
        M = self.data.M
        nle = self.data.nle

        # Reference acceleration
        qr_dot = self.dq_des + self.lambda_val * e
        qr_ddot = self.ddq_des + self.lambda_val * de

        # Equivalent control (nominal)
        tau_eq = M @ qr_ddot + nle

        # Robustness term
        tau_robust = self.K * self.sat(s)

        return tau_eq + tau_robust


# ============================================================================
# PART 3: MAIN SIMULATION FUNCTION
# ============================================================================

def run_final_project():

    print("=" * 70)
    print("FINAL PROJECT: Robust Sliding Mode Control for Robot Manipulators")
    print("=" * 70)

    # Create directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    Path("logs/plots").mkdir(parents=True, exist_ok=True)

    # Path to robot model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")

    # ========================================================================
    # Task 1: Inverse Dynamics Controller
    # ========================================================================
    print("\n" + "-" * 70)
    print("TASK 1: Inverse Dynamics Controller (40 points)")
    print("-" * 70)

    sim_id = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=False,
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/06_id_controller.mp4",
        width=1920,
        height=1080
    )

    # Add uncertainties as specified
    damping = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
    friction = np.array([1.5, 0.5, 0.5, 0.1, 0.1, 0.1])
    sim_id.set_joint_damping(damping)
    sim_id.set_joint_friction(friction)
    sim_id.modify_body_properties("end_effector", mass=4.0)

    id_controller = InverseDynamicsController(urdf_path)

    # Data collection for ID
    id_data = {'time': [], 'q': [], 'error': [], 'tau': []}
    t_id = 0

    def id_wrapper(q, dq, t):
        tau = id_controller.compute_torque(q, dq)
        id_data['time'].append(t)
        id_data['q'].append(q.copy())
        id_data['error'].append(id_controller.q_des - q)
        id_data['tau'].append(tau.copy())
        return tau

    sim_id.set_controller(id_wrapper)
    sim_id.run(time_limit=5.0)

    # Convert to arrays
    for key in id_data:
        id_data[key] = np.array(id_data[key])

    # ========================================================================
    # Task 2: Sliding Mode Controller
    # ========================================================================
    print("\n" + "-" * 70)
    print("TASK 2: Sliding Mode Controller (40 points)")
    print("-" * 70)

    # Test with boundary layer
    print("\n2A: SMC with Boundary Layer (φ=0.1)")
    sim_smc_bl = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=False,
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/06_smc_with_bl.mp4",
        width=1920,
        height=1080
    )

    # Same uncertainties
    sim_smc_bl.set_joint_damping(damping)
    sim_smc_bl.set_joint_friction(friction)
    sim_smc_bl.modify_body_properties("end_effector", mass=4.0)

    smc_bl = SlidingModeController(urdf_path, use_bl=True, phi=0.1)

    smc_bl_data = {'time': [], 'q': [], 'error': [], 'tau': [], 's': []}

    def smc_bl_wrapper(q, dq, t):
        tau = smc_bl.compute_torque(q, dq)
        smc_bl_data['time'].append(t)
        smc_bl_data['q'].append(q.copy())
        smc_bl_data['error'].append(smc_bl.q_des - q)
        smc_bl_data['tau'].append(tau.copy())

        # Sliding surface
        e = smc_bl.q_des - q
        de = smc_bl.dq_des - dq
        s = de + smc_bl.lambda_val * e
        smc_bl_data['s'].append(s.copy())

        return tau

    sim_smc_bl.set_controller(smc_bl_wrapper)
    sim_smc_bl.run(time_limit=5.0)

    for key in smc_bl_data:
        smc_bl_data[key] = np.array(smc_bl_data[key])

    # Test without boundary layer (for chattering analysis)
    print("\n2B: SMC without Boundary Layer (for chattering analysis)")
    sim_smc_nobl = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=False,
        show_viewer=False,  # No viewer for faster simulation
        record_video=True,
        video_path="logs/videos/06_smc_no_bl.mp4",
        width=1920,
        height=1080
    )

    sim_smc_nobl.set_joint_damping(damping)
    sim_smc_nobl.set_joint_friction(friction)
    sim_smc_nobl.modify_body_properties("end_effector", mass=4.0)

    smc_nobl = SlidingModeController(urdf_path, use_bl=False)

    smc_nobl_data = {'time': [], 'tau': []}

    def smc_nobl_wrapper(q, dq, t):
        tau = smc_nobl.compute_torque(q, dq)
        smc_nobl_data['time'].append(t)
        smc_nobl_data['tau'].append(tau.copy())
        return tau

    sim_smc_nobl.set_controller(smc_nobl_wrapper)
    sim_smc_nobl.run(time_limit=2.0)  # Shorter due to chattering

    for key in smc_nobl_data:
        smc_nobl_data[key] = np.array(smc_nobl_data[key])

    # ========================================================================
    # Task 3: Boundary Layer Analysis (20 points)
    # ========================================================================
    print("\n" + "-" * 70)
    print("TASK 3: Boundary Layer Analysis (20 points)")
    print("-" * 70)

    # Test different boundary layer thicknesses
    phi_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    bl_results = []

    for phi in phi_values:
        print(f"\nTesting φ = {phi}")

        # Quick simulation
        sim_test = Simulator(
            xml_path="robots/universal_robots_ur5e/scene.xml",
            enable_task_space=False,
            show_viewer=False,
            record_video=False
        )

        sim_test.set_joint_damping(damping)
        sim_test.set_joint_friction(friction)
        sim_test.modify_body_properties("end_effector", mass=4.0)

        controller = SlidingModeController(urdf_path, use_bl=True, phi=phi)

        test_data = {'error': [], 'tau': []}

        def test_wrapper(q, dq, t):
            tau = controller.compute_torque(q, dq)
            test_data['error'].append(controller.q_des - q)
            test_data['tau'].append(tau.copy())
            return tau

        sim_test.set_controller(test_wrapper)
        sim_test.reset()

        # Run brief simulation
        for i in range(1000):
            state = sim_test.get_state()
            tau = test_wrapper(state['q'], state['dq'], i * 0.002)
            sim_test.step(tau)

        # Compute metrics
        error = np.array(test_data['error'])
        tau = np.array(test_data['tau'])

        rmse = np.mean(np.sqrt(np.mean(error ** 2, axis=0)))
        torque_var = np.mean(np.var(tau, axis=0))

        bl_results.append({
            'phi': phi,
            'rmse': rmse,
            'torque_variance': torque_var
        })

        print(f"  RMSE: {rmse:.6f}, Torque variance: {torque_var:.4f}")

    # ========================================================================
    # Generate Plots and Analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("GENERATING RESULTS AND PLOTS")
    print("-" * 70)

    # Plot 1: Comparison of tracking performance
    plt.figure(figsize=(12, 8))

    # Joint 1 errors for all controllers
    plt.subplot(2, 2, 1)
    plt.plot(id_data['time'], id_data['error'][:, 0], 'b-', label='ID', linewidth=2)
    plt.plot(smc_bl_data['time'], smc_bl_data['error'][:, 0], 'g-', label='SMC (BL)', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint 1 Error [rad]')
    plt.title('Joint 1 Tracking Error')
    plt.legend()
    plt.grid(True)

    # Joint 2 errors
    plt.subplot(2, 2, 2)
    plt.plot(id_data['time'], id_data['error'][:, 1], 'b-', label='ID', linewidth=2)
    plt.plot(smc_bl_data['time'], smc_bl_data['error'][:, 1], 'g-', label='SMC (BL)', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint 2 Error [rad]')
    plt.title('Joint 2 Tracking Error')
    plt.legend()
    plt.grid(True)

    # Control torques comparison
    plt.subplot(2, 2, 3)
    plt.plot(id_data['time'][:500], id_data['tau'][:500, 0], 'b-', label='ID', linewidth=1, alpha=0.7)
    plt.plot(smc_bl_data['time'][:500], smc_bl_data['tau'][:500, 0], 'g-', label='SMC (BL)', linewidth=1, alpha=0.7)
    plt.plot(smc_nobl_data['time'][:500], smc_nobl_data['tau'][:500, 0], 'r-', label='SMC (no BL)', linewidth=1,
             alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint 1 Torque [Nm]')
    plt.title('Control Torque Comparison')
    plt.legend()
    plt.grid(True)

    # Boundary layer analysis
    plt.subplot(2, 2, 4)
    phi_vals = [r['phi'] for r in bl_results]
    rmse_vals = [r['rmse'] for r in bl_results]
    var_vals = [r['torque_variance'] for r in bl_results]

    ax1 = plt.gca()
    ax1.plot(phi_vals, rmse_vals, 'bo-', linewidth=2, markersize=8, label='RMSE')
    ax1.set_xlabel('Boundary Layer Thickness φ')
    ax1.set_ylabel('RMSE [rad]', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xscale('log')

    ax2 = ax1.twinx()
    ax2.plot(phi_vals, var_vals, 'ro-', linewidth=2, markersize=8, label='Torque Variance')
    ax2.set_ylabel('Torque Variance [Nm²]', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Boundary Layer Trade-off')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('logs/plots/06_final_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ========================================================================
    # Print Summary Results
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY RESULTS")
    print("=" * 70)

    # Compute metrics
    def compute_performance_metrics(data, name):
        rmse = np.mean(np.sqrt(np.mean(data['error'] ** 2, axis=0)))
        max_error = np.max(np.abs(data['error']))
        mean_torque = np.mean(np.abs(data['tau']))
        return rmse, max_error, mean_torque

    id_rmse, id_max, id_torque = compute_performance_metrics(id_data, "ID")
    smc_rmse, smc_max, smc_torque = compute_performance_metrics(smc_bl_data, "SMC")

    print(f"\nInverse Dynamics Controller:")
    print(f"  RMSE: {id_rmse:.6f} rad")
    print(f"  Max Error: {id_max:.6f} rad")
    print(f"  Mean Torque: {id_torque:.2f} Nm")

    print(f"\nSliding Mode Controller (with BL):")
    print(f"  RMSE: {smc_rmse:.6f} rad")
    print(f"  Max Error: {smc_max:.6f} rad")
    print(f"  Mean Torque: {smc_torque:.2f} Nm")

    improvement = (id_rmse - smc_rmse) / id_rmse * 100
    print(f"\nImprovement with SMC: {improvement:+.1f}% in RMSE")

    # Chattering analysis
    chatter_index = np.mean(np.var(smc_nobl_data['tau'], axis=0))
    smooth_index = np.mean(np.var(smc_bl_data['tau'], axis=0))
    print(f"\nChattering Analysis:")
    print(f"  Torque variance (no BL): {chatter_index:.4f} Nm²")
    print(f"  Torque variance (with BL): {smooth_index:.4f} Nm²")
    print(f"  Chattering reduction: {(chatter_index - smooth_index) / chatter_index * 100:.1f}%")

    print(f"\nBoundary Layer Analysis:")
    for result in bl_results:
        print(f"  φ={result['phi']}: RMSE={result['rmse']:.6f}, "
              f"Variance={result['torque_variance']:.4f}")

    print("\n" + "=" * 70)
    print("FINAL OBSERVATIONS:")
    print("=" * 70)
    print("1. Inverse Dynamics: Sensitive to model uncertainties")
    print("2. Sliding Mode Control: Robust to uncertainties")
    print("3. Boundary Layer: Essential for practical implementation")
    print("4. Optimal φ: 0.1 provides good robustness-smoothness trade-off")
    print("\nCheck 'logs/videos/' for simulation videos")
    print("Check 'logs/plots/' for performance plots")


if __name__ == "__main__":
    run_final_project()