"""
OCTO-MYCELIAL NEUROMORPHIC SENSORY DEFENSE SYSTEM v4.1
Enhanced with RISS Scoring, Threat Taxonomy & LUMINARK Integration

Integration of:
- Octo-Mycelial v4.0 (biological sensory capabilities)
- RISS (Recursive Impact & State Score) — SAP-aware threat scoring
- Cyber Kill Chain threat taxonomy
- Network variability as direct HRV analog
- LUMINARK Ma'at ethical audit bridge

Fixes in v4.1:
✓ SAR → SAP terminology corrected throughout
✓ SAP Stage 3: "first self-reflection, 3D complexity" (corrected)
✓ SAP Stage 8: "false hell, mastering duality/gratitude releases polarity" (corrected)
✓ scipy.signal.ricker removed in scipy >= 1.4 — replaced with manual Ricker wavelet
✓ fuse_sensors() hardcoded num_nodes=100 → uses actual network size
✓ collective_awareness NaN crash fixed
✓ LUMINARK Ma'at ethical audit bridge added
✓ All original v4.0 capabilities preserved

Run: python -m luminark.octo_mycelial --mode demo
"""

import asyncio
import json
import time
import random
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Any
from enum import Enum
from functools import lru_cache

import numpy as np
import pandas as pd
import networkx as nx
from scipy import signal, fft
from scipy.spatial import KDTree
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# ============================================================================
# ENUMS & DATA STRUCTURES
# ============================================================================

class PolyvagalState(Enum):
    VENTRAL_RENEWAL = "ventral renewal"
    SYMPATHETIC_THRESHOLD = "sympathetic threshold"
    DORSAL_TRAP = "dorsal trap"

class ChipState(Enum):
    VENTRAL_RENEWAL = "ventral renewal"
    SYMPATHETIC_THRESHOLD = "sympathetic threshold"
    DORSAL_TRAP = "dorsal trap"

class EthicalPrinciple(Enum):
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    HUMAN_OVERRIDE = "human_override"

class ThreatType(Enum):
    """Cyber Kill Chain threat taxonomy"""
    RECON = "reconnaissance"
    LATERAL_MOVEMENT = "lateral_movement"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"
    COMMAND_CONTROL = "command_and_control"

@dataclass
class PhysiologicalMetrics:
    """Container for normalized physiological metrics"""
    hrv_score: float           # 0-100
    sleep_score: float         # 0-100
    resp_score: float          # 0-100
    o2_score: float            # 0-100
    combined: float            # 0-100
    network_variability: float # HRV analog for network (0-100)
    timestamp: datetime

    def to_dict(self):
        return {
            'hrv': self.hrv_score,
            'sleep': self.sleep_score,
            'resp': self.resp_score,
            'o2': self.o2_score,
            'combined': self.combined,
            'network_variability': self.network_variability,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class ThreatEvent:
    """Individual threat event with RISS scoring"""
    node_id: int
    threat_type: ThreatType
    timestamp: float
    riss_score: int    # Recursive Impact & State Score (0-100)
    severity: str
    sap_stage: int     # SAP stage at time of threat
    contained: bool = False

    def to_dict(self):
        return {
            'node_id': self.node_id,
            'threat_type': self.threat_type.value,
            'timestamp': self.timestamp,
            'riss_score': self.riss_score,
            'severity': self.severity,
            'sap_stage': self.sap_stage,
            'contained': self.contained
        }

# ============================================================================
# RISS CALCULATOR (SAP-AWARE THREAT SCORING)
# FIX v4.1: Corrected all "SAR" references to "SAP"
# FIX v4.1: Stage 3 and Stage 8 descriptions corrected to match Richard Stanfield's SAP Framework
# ============================================================================

class RISSCalculator:
    """
    Recursive Impact & State Score Calculator
    Integrates SAP (Stages of Awareness and Perception) stage weights
    with cyber threat assessment.

    SAP Stage Reference (Richard Stanfield's Framework):
        Stage 1 — SEED:       Initial dormancy, minimal awareness
        Stage 2 — ROOT:       Building foundations, emerging structure
        Stage 3 — FLUID:      First self-reflection + 3D complexity (initial compromise risk)
        Stage 4 — FOUNDATION: Established presence, grounded complexity
        Stage 5 — THRESHOLD:  Critical decision point — highest volatility
        Stage 6 — INTEGRATION: Peak complexity, multiple vectors active
        Stage 7 — DISTILLATION: Analysis phase, pattern recognition
        Stage 8 — FALSE HELL: Maximum danger — mastering duality; gratitude releases
                               polarity and opens the path toward Stage 9
        Stage 9 — TEACHING:   Transformation, integration, recovery
    """

    def __init__(self):
        # SAP stage weights — threat amplification per stage
        self.stage_weights = {
            1: 0.3,   # SEED — dormant, low impact
            2: 0.5,   # ROOT — building
            3: 0.7,   # FLUID — first self-reflection, 3D complexity
            4: 1.0,   # FOUNDATION — established presence
            5: 1.8,   # THRESHOLD — critical decision point (highest volatility)
            6: 1.5,   # INTEGRATION — peak complexity
            7: 1.6,   # DISTILLATION — analysis / pattern recognition
            8: 2.2,   # FALSE HELL — maximum danger; duality mastery is the exit
            9: 1.4    # TEACHING — recovery / transformation
        }

        # Threat type base scores (Cyber Kill Chain)
        self.threat_base_scores = {
            ThreatType.RECON: 30,
            ThreatType.LATERAL_MOVEMENT: 50,
            ThreatType.PERSISTENCE: 70,
            ThreatType.PRIVILEGE_ESCALATION: 65,
            ThreatType.COMMAND_CONTROL: 75,
            ThreatType.EXFILTRATION: 85,
            ThreatType.IMPACT: 90
        }

    def calculate_riss(self,
                       threat_type: ThreatType,
                       affected_nodes: int,
                       network_variability: float,
                       sap_stage: int = 5) -> int:
        """
        Calculate Recursive Impact & State Score

        Formula:
            RISS = (base_score + scale_factor + variability_penalty) * stage_weight

        Where:
            base_score          = threat type severity (30-90)
            scale_factor        = network spread impact (0-50)
            variability_penalty = HRV degradation penalty (0-30)
            stage_weight        = SAP stage multiplier (0.3-2.2)
        """
        base = self.threat_base_scores.get(threat_type, 50)
        scale = min(50, affected_nodes * 3)
        variability_penalty = (100 - network_variability) * 0.3
        stage_multiplier = self.stage_weights.get(sap_stage, 1.0)

        raw_riss = (base + scale + variability_penalty) * stage_multiplier
        return min(100, int(raw_riss))

    def determine_sap_stage(self,
                            threat_type: ThreatType,
                            network_health: float) -> int:
        """
        Map threat progression to SAP stages.

        Stage 1  — Reconnaissance (initial probing)
        Stage 3  — Initial compromise (first self-reflection, 3D complexity)
        Stage 4  — Established presence (foundation)
        Stage 5  — Lateral movement (threshold — critical decision point)
        Stage 6  — Peak complexity (multiple attack vectors active)
        Stage 7  — Analysis/distillation phase
        Stage 8  — Critical infrastructure impact (false hell — maximum danger)
        Stage 9  — Recovery / teaching phase
        """
        threat_stage_map = {
            ThreatType.RECON: 1,
            ThreatType.LATERAL_MOVEMENT: 5,   # THRESHOLD
            ThreatType.PERSISTENCE: 4,
            ThreatType.PRIVILEGE_ESCALATION: 6,
            ThreatType.COMMAND_CONTROL: 7,
            ThreatType.EXFILTRATION: 7,
            ThreatType.IMPACT: 8              # FALSE HELL
        }

        base_stage = threat_stage_map.get(threat_type, 5)

        # Adjust based on network health
        if network_health < 30:
            return 8   # System in crisis — Stage 8 (false hell)
        elif network_health < 50:
            return min(7, base_stage + 1)   # Degraded — Stage 5-7 range
        else:
            return base_stage               # Healthy — maintain base stage

# ============================================================================
# MYCELIUM SENSORY SYSTEM (Armillaria ostoyae)
# ============================================================================

def _ricker_wavelet(points: int, a: float) -> np.ndarray:
    """
    Manual Ricker (Mexican hat) wavelet.
    FIX v4.1: Replaces scipy.signal.ricker which was removed in scipy >= 1.4.

    Args:
        points: Number of points in the wavelet
        a: Width parameter (scale)
    Returns:
        Normalized Ricker wavelet array
    """
    A = 2 / (np.sqrt(3 * a) * (np.pi ** 0.25))
    wsq = a ** 2
    vec = np.arange(0, points) - (points - 1.0) / 2
    tsq = vec ** 2
    mod = 1 - tsq / wsq
    gauss = np.exp(-tsq / (2 * wsq))
    return A * mod * gauss


def _cwt_ricker(data: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """
    Continuous Wavelet Transform using manual Ricker wavelet.
    FIX v4.1: Replaces scipy.signal.cwt(data, signal.ricker, widths) which crashes
    on scipy >= 1.4 because signal.ricker was removed.
    """
    out = np.zeros((len(widths), len(data)))
    for i, width in enumerate(widths):
        wavelet = _ricker_wavelet(min(10 * int(width), len(data)), width)
        out[i] = np.convolve(data, wavelet, mode='same')
    return out


class MyceliumSensorySystem:
    """World's Largest Mycelium (2,400 acres, 2,500 years) Sensory Capabilities"""

    def __init__(self, network_size: int):
        self.network_size = network_size
        self.conductivity = 0.85         # Biological tissue conductivity (S/m)
        self.signal_velocity = 0.5       # m/s in fungal hyphae
        self.resonance_frequencies = [7, 14, 28, 42]  # Hz — mycelial network resonance

    def detect_chemical_gradient(self, node_positions: np.ndarray,
                                 threat_chemicals: np.ndarray) -> np.ndarray:
        """Mycelium detects chemical gradients (calcium, potassium, pH changes)"""
        gradient = np.zeros((len(node_positions), threat_chemicals.shape[1]
                             if threat_chemicals.ndim > 1 else 1))
        for i, pos in enumerate(node_positions):
            distances = np.linalg.norm(node_positions - pos, axis=1)
            attenuation = np.exp(-distances / 10.0)
            if threat_chemicals.ndim > 1:
                chemical_field = np.sum(threat_chemicals * attenuation[:, np.newaxis], axis=0)
            else:
                chemical_field = np.array([np.sum(threat_chemicals * attenuation)])
            gradient[i] = chemical_field
        return gradient

    def sense_electrical_patterns(self, node_activity: np.ndarray) -> Dict:
        """Mycelium conducts electrical signals, detects energy surges"""
        if len(node_activity) < 2:
            return {'resonance_frequencies': [], 'energy_surges': [],
                    'total_power': 0.0, 'dominant_frequency': 0.0}

        frequencies = fft.fftfreq(len(node_activity), 0.01)
        power_spectrum = np.abs(fft.fft(node_activity)) ** 2

        resonance_detected = []
        for freq in self.resonance_frequencies:
            idx = np.argmin(np.abs(frequencies - freq))
            if power_spectrum[idx] > np.mean(power_spectrum) * 3:
                resonance_detected.append(freq)

        mean_activity = np.mean(node_activity)
        std_activity = np.std(node_activity)
        if std_activity > 0:
            surge_mask = node_activity > mean_activity + 3 * std_activity
        else:
            surge_mask = np.zeros(len(node_activity), dtype=bool)

        return {
            'resonance_frequencies': resonance_detected,
            'energy_surges': np.where(surge_mask)[0].tolist(),
            'total_power': float(np.sum(power_spectrum)),
            'dominant_frequency': float(frequencies[np.argmax(power_spectrum)])
        }

    def detect_vibrations(self, node_movements: np.ndarray) -> Dict:
        """
        Mycelium senses soil vibrations (0.1-100 Hz biological range).
        FIX v4.1: Replaced scipy.signal.cwt + signal.ricker (removed in scipy >= 1.4)
                  with _cwt_ricker() manual implementation.
        """
        if len(node_movements) < 2:
            return {'vibration_intensity': 0.0, 'rhythmic_patterns': None,
                    'vibration_map': np.array([]), 'anomalous_vibrations': []}

        vibrations = np.zeros(len(node_movements))
        for i, movement in enumerate(node_movements):
            # Scalar movement → expand to 1-D signal for CWT
            if np.isscalar(movement):
                signal_1d = np.full(max(32, len(node_movements)), float(movement))
            else:
                signal_1d = np.asarray(movement, dtype=float)

            coefficients = _cwt_ricker(signal_1d, np.arange(1, 31))
            dominant_vibration = np.mean(np.abs(coefficients[5:25]))
            vibrations[i] = dominant_vibration

        autocorr = np.correlate(node_movements, node_movements, mode='full')
        mid = len(node_movements) // 2
        periodicity = int(np.argmax(autocorr[mid + 1:]) + 1)

        mean_vib = np.mean(vibrations)
        anomalous = (np.where(vibrations > mean_vib * 2)[0].tolist()
                     if mean_vib > 0 else [])

        return {
            'vibration_intensity': float(np.mean(vibrations)),
            'rhythmic_patterns': periodicity if periodicity < mid else None,
            'vibration_map': vibrations,
            'anomalous_vibrations': anomalous
        }

    def sense_mineral_concentrations(self, node_health: np.ndarray) -> Dict:
        """Armillaria detects mineral imbalances (Ca²⁺, K⁺, Mg²⁺, Fe³⁺)"""
        calcium   = node_health * 0.7 + np.random.normal(0, 0.1, len(node_health))
        potassium = node_health * 0.5 + np.random.normal(0, 0.08, len(node_health))
        magnesium = node_health * 0.3 + np.random.normal(0, 0.05, len(node_health))

        return {
            'calcium_deficit':   np.where(calcium < 0.4)[0].tolist(),
            'potassium_deficit': np.where(potassium < 0.3)[0].tolist(),
            'magnesium_deficit': np.where(magnesium < 0.2)[0].tolist()
        }

# ============================================================================
# OCTOPUS SENSORY SYSTEM (Cephalopoda)
# ============================================================================

class OctopusSensorySystem:
    """Octopus Sensory Capabilities (500M neurons, distributed intelligence)"""

    def __init__(self):
        self.polarization_angles = np.linspace(0, 180, 36)
        self.camouflage_patterns = {
            'mimicry':        {'frequency': 0.3, 'complexity': 0.8},
            'disruptive':     {'frequency': 0.5, 'complexity': 0.6},
            'countershading': {'frequency': 0.2, 'complexity': 0.4}
        }
        self.chromatophore_states = {}
        self.sucker_chemical_memory = {}

    def polarized_light_vision(self, light_field: np.ndarray) -> Dict:
        """Octopus sees polarized light patterns (invisible to humans)"""
        if light_field is None or len(light_field) == 0:
            return {}

        polarization_vectors = np.array([
            light_field * np.cos(np.deg2rad(angle))
            for angle in self.polarization_angles
        ])

        polarization_entropy = -np.sum(
            polarization_vectors * np.log2(polarization_vectors + 1e-10), axis=0
        )

        anomalies = np.where(
            polarization_entropy > np.mean(polarization_entropy) * 1.5
        )[0]

        return {
            'polarization_pattern': polarization_vectors,
            'polarization_entropy': polarization_entropy,
            'anomaly_indices': anomalies.tolist(),
            'pattern_complexity': float(np.std(polarization_entropy))
        }

    def chemotactile_sensing(self, node_positions: np.ndarray,
                             chemical_signatures: Dict) -> Dict:
        """Octopus suckers taste what they touch (10,000+ receptors per sucker)"""
        chemical_detections = {}
        for node_id, position in enumerate(node_positions):
            detections = []
            for chem_name, chem_field in chemical_signatures.items():
                if isinstance(chem_field, dict):
                    concentration = chem_field.get(node_id, 0)
                elif hasattr(chem_field, '__len__') and node_id < len(chem_field):
                    concentration = float(np.mean(np.abs(chem_field[node_id])))
                else:
                    concentration = 0
                if concentration > 0.1:
                    detections.append({
                        'chemical': chem_name,
                        'concentration': float(concentration),
                        'novelty': 1.0
                    })
            if detections:
                chemical_detections[node_id] = detections
        return chemical_detections

    def proprioceptive_awareness(self, node_positions: np.ndarray,
                                 node_velocities: np.ndarray) -> Dict:
        """
        Octopus knows arm positions without looking (distributed processing).
        FIX v4.1: collective_awareness NaN crash fixed — handles empty synchrony arrays.
        """
        if len(node_positions) == 0:
            return {
                'position_uncertainty': np.array([]),
                'movement_synchrony': np.array([]),
                'proprioceptive_anomalies': [],
                'collective_awareness': 0.0
            }

        position_uncertainty = np.zeros(len(node_positions))
        movement_synchrony = np.zeros((len(node_positions), len(node_positions)))

        for i in range(len(node_positions)):
            distance_from_center = np.linalg.norm(node_positions[i])
            position_uncertainty[i] = distance_from_center * 0.1

            for j in range(len(node_positions)):
                if i != j:
                    vel_i = (node_velocities[i]
                             if i < len(node_velocities) else np.zeros(2))
                    vel_j = (node_velocities[j]
                             if j < len(node_velocities) else np.zeros(2))
                    norm_i = np.linalg.norm(vel_i)
                    norm_j = np.linalg.norm(vel_j)
                    if norm_i > 0 and norm_j > 0:
                        movement_synchrony[i, j] = (
                            np.dot(vel_i, vel_j) / (norm_i * norm_j)
                        )

        proprioceptive_anomalies = np.where(position_uncertainty > 0.5)[0]

        # FIX v4.1: was np.mean(movement_synchrony[movement_synchrony > 0])
        # which returns NaN when no positive values exist → now returns 0.0 safely
        positive_sync = movement_synchrony[movement_synchrony > 0]
        collective_awareness = float(np.mean(positive_sync)) if len(positive_sync) > 0 else 0.0

        return {
            'position_uncertainty': position_uncertainty,
            'movement_synchrony': movement_synchrony,
            'proprioceptive_anomalies': proprioceptive_anomalies.tolist(),
            'collective_awareness': collective_awareness
        }

# ============================================================================
# THERMAL & ENERGY SENSING
# ============================================================================

class ThermalEnergySensing:
    """Detects heat signatures and energy surges (combined biological sensing)"""

    def __init__(self):
        self.thermal_baseline = None
        self.energy_baseline = None
        self.thermal_history = []
        self.energy_history = []

    def detect_thermal_anomalies(self, node_temperatures: np.ndarray,
                                 ambient_temperature: float) -> Dict:
        """Detect thermal anomalies (heat signatures)"""
        if self.thermal_baseline is None:
            self.thermal_baseline = float(np.mean(node_temperatures))

        temp_anomalies = node_temperatures - ambient_temperature
        thermal_anomaly_nodes = np.where(np.abs(temp_anomalies) > 2.0)[0]
        thermal_gradient = np.gradient(node_temperatures)
        high_gradient_nodes = np.where(np.abs(thermal_gradient) > 1.0)[0]

        return {
            'thermal_anomalies': thermal_anomaly_nodes.tolist(),
            'thermal_gradients': thermal_gradient.tolist(),
            'high_gradient_nodes': high_gradient_nodes.tolist(),
            'ambient_temperature': ambient_temperature
        }

    def detect_energy_surges(self, node_energy: np.ndarray) -> Dict:
        """Detect energy surges and power anomalies"""
        if self.energy_baseline is None:
            self.energy_baseline = float(np.mean(node_energy))

        self.energy_history.append(node_energy.copy())
        if len(self.energy_history) > 10:
            self.energy_history = self.energy_history[-10:]

        if len(self.energy_history) > 1:
            energy_change = node_energy - self.energy_history[-2]
            std_e = np.std(node_energy)
            surge_nodes = (np.where(energy_change > std_e * 3)[0]
                          if std_e > 0 else np.array([]))
        else:
            surge_nodes = np.array([])

        return {
            'energy_surges': surge_nodes.tolist(),
            'total_energy': float(np.sum(node_energy)),
            'energy_variance': float(np.var(node_energy))
        }

# ============================================================================
# BIO-SENSORY FUSION
# ============================================================================

class BioSensoryFusion:
    """Fuses mycelium and octopus sensory capabilities with attention mechanism"""

    def __init__(self, network_size: int):
        self.network_size = network_size
        self.mycelium_sensors = MyceliumSensorySystem(network_size)
        self.octopus_sensors = OctopusSensorySystem()
        self.attention_weights = {
            'vibration':      0.25,
            'chemical':       0.20,
            'electrical':     0.15,
            'visual':         0.20,
            'proprioceptive': 0.10,
            'thermal':        0.10
        }
        self.calibration_history = []

    def sense_environment(self, network_state: Dict) -> Dict:
        """Comprehensive environmental sensing using all biological modalities"""
        sensory_data = {}

        node_positions    = network_state.get('node_positions', np.array([]))
        node_health       = network_state.get('node_health', np.array([]))
        node_activity     = network_state.get('node_activity', np.array([]))
        threat_signatures = network_state.get('threat_signatures', {})

        if len(node_positions) > 0:
            if 'chemical_signatures' in threat_signatures:
                chemical_gradients = self.mycelium_sensors.detect_chemical_gradient(
                    node_positions, threat_signatures['chemical_signatures']
                )
                sensory_data['chemical_gradients'] = chemical_gradients

            sensory_data['electrical_patterns'] = (
                self.mycelium_sensors.sense_electrical_patterns(node_activity)
            )
            sensory_data['vibrations'] = (
                self.mycelium_sensors.detect_vibrations(node_activity)
            )
            sensory_data['mineral_deficiencies'] = (
                self.mycelium_sensors.sense_mineral_concentrations(node_health)
            )

        if 'light_field' in network_state:
            sensory_data['polarized_vision'] = (
                self.octopus_sensors.polarized_light_vision(network_state['light_field'])
            )

        sensory_data['chemotactile_detections'] = (
            self.octopus_sensors.chemotactile_sensing(node_positions, threat_signatures)
        )

        sensory_data['proprioceptive_awareness'] = (
            self.octopus_sensors.proprioceptive_awareness(
                node_positions,
                network_state.get('node_velocities', np.array([]))
            )
        )

        sensory_data['fused_threat_assessment'] = self.fuse_sensors(
            sensory_data, actual_network_size=len(node_positions)
        )
        return sensory_data

    def fuse_sensors(self, sensory_data: Dict,
                     actual_network_size: int = 0) -> Dict:
        """
        Sensor fusion using attention-weighted integration.
        FIX v4.1: Was hardcoded num_nodes=100 — now uses actual_network_size
                  so threat scores reflect the real network topology.
        """
        # FIX: use actual network size, not a hardcoded constant
        num_nodes = actual_network_size if actual_network_size > 0 else self.network_size
        threat_scores = {i: 0.0 for i in range(num_nodes)}

        # Vibration-based threat scoring
        if 'vibrations' in sensory_data:
            vibrations = sensory_data['vibrations']
            vibration_threat = vibrations.get('vibration_map', np.zeros(num_nodes))
            for i in range(min(num_nodes, len(vibration_threat))):
                threat_scores[i] += (float(vibration_threat[i]) *
                                     self.attention_weights['vibration'])

        # Electrical pattern scoring
        if 'electrical_patterns' in sensory_data:
            surges = sensory_data['electrical_patterns'].get('energy_surges', [])
            for node in surges:
                if node < num_nodes:
                    threat_scores[node] += self.attention_weights['electrical'] * 2.0

        # Chemical gradient scoring
        if 'chemotactile_detections' in sensory_data:
            detections = sensory_data['chemotactile_detections']
            for node_id, det_list in detections.items():
                if node_id < num_nodes and det_list:
                    max_conc = max(d['concentration'] for d in det_list)
                    threat_scores[node_id] += max_conc * self.attention_weights['chemical']

        # Proprioceptive anomaly scoring
        if 'proprioceptive_awareness' in sensory_data:
            anomalies = sensory_data['proprioceptive_awareness'].get(
                'proprioceptive_anomalies', []
            )
            for node in anomalies:
                if node < num_nodes:
                    threat_scores[node] += self.attention_weights['proprioceptive']

        # Normalize threat scores
        max_threat = max(threat_scores.values()) if threat_scores else 1.0
        if max_threat > 0:
            threat_scores = {k: v / max_threat for k, v in threat_scores.items()}

        # Categorize threat levels
        threat_categories = {}
        for node, score in threat_scores.items():
            if   score > 0.8: threat_categories[node] = 'CRITICAL'
            elif score > 0.6: threat_categories[node] = 'HIGH'
            elif score > 0.4: threat_categories[node] = 'MEDIUM'
            elif score > 0.2: threat_categories[node] = 'LOW'
            else:             threat_categories[node] = 'NORMAL'

        return {
            'threat_scores': threat_scores,
            'threat_categories': threat_categories,
            'overall_threat_level': (
                float(np.mean(list(threat_scores.values()))) if threat_scores else 0.0
            )
        }

# ============================================================================
# PHYSIOLOGICAL DATA PIPELINE
# ============================================================================

class PhysiologicalPipeline:
    """Processes and normalizes physiological data from multiple sources"""

    def __init__(self):
        self.cache = {}
        self.cache_ttl = 5

    def load_csv_safe(self, path: Optional[str], column: str) -> List[float]:
        """Safely load CSV data with validation"""
        if not path or not os.path.exists(path):
            return []
        try:
            df = pd.read_csv(path)
            if column not in df.columns:
                return []
            return df[column].astype(float).dropna().tolist()
        except Exception:
            return []

    def normalize_hrv(self, hrv_ms: float) -> float:
        """Normalize HRV (20-100 ms range) to 0-100 scale"""
        return max(0.0, min(100.0, ((hrv_ms - 20) / 80) * 100))

    def compute_sleep_score(self, duration_min: float, deep_pct: float) -> float:
        """Calculate sleep quality score (0-100)"""
        duration_score = min(100.0, (duration_min / 480) * 100)
        deep_score = min(100.0, (deep_pct / 25) * 100)
        return duration_score * 0.6 + deep_score * 0.4

    def compute_resp_score(self, resp_rate: float) -> float:
        """Calculate respiratory stability score (0-100)"""
        if 12 <= resp_rate <= 20:
            return 100.0
        elif resp_rate < 12:
            return max(0.0, (resp_rate / 12) * 100)
        else:
            return max(0.0, 100.0 - (resp_rate - 20) * 10)

    def compute_o2_score(self, o2_percent: float) -> float:
        """Calculate O2 saturation score (0-100)"""
        if o2_percent >= 95:
            return min(100.0, ((o2_percent - 95) / 5) * 100)
        else:
            return max(0.0, ((o2_percent - 80) / 15) * 100)

# ============================================================================
# OCTO-MYCELIAL CORE DEFENSE v4.1
# ============================================================================

class OctoMycelialChip:
    """
    Enhanced Octo-Mycelial defense system v4.1

    NEW in v4.1:
    - All SAR references corrected to SAP (Stages of Awareness & Perception)
    - Stage 3 and Stage 8 descriptions corrected per Richard Stanfield's SAP Framework
    - scipy.signal.ricker crash fixed (manual Ricker wavelet implementation)
    - fuse_sensors() now uses actual network size instead of hardcoded 100
    - collective_awareness NaN crash fixed
    - LUMINARK Ma'at ethical audit bridge (see audit_threat_event())
    """

    def __init__(self,
                 num_nodes: int = 35,
                 hrv_csv_path: Optional[str] = None,
                 sleep_csv_path: Optional[str] = None,
                 resp_csv_path: Optional[str] = None,
                 o2_csv_path: Optional[str] = None):

        self.G = nx.random_geometric_graph(num_nodes, radius=0.25)
        self._init_node_states()

        self.bio_fusion     = BioSensoryFusion(num_nodes)
        self.thermal_energy = ThermalEnergySensing()
        self.physio_pipeline = PhysiologicalPipeline()
        self.riss_calculator = RISSCalculator()

        self.threat_history: List[ThreatEvent] = []
        self.network_variability = 75.0   # Starts healthy

        self.hrv_data   = self.physio_pipeline.load_csv_safe(hrv_csv_path, 'value')
        self.resp_data  = self.physio_pipeline.load_csv_safe(resp_csv_path, 'breaths_per_minute')
        self.o2_data    = self.physio_pipeline.load_csv_safe(o2_csv_path, 'o2_percentage')
        self.sleep_data = self._load_sleep_data(sleep_csv_path)

        self.threats       = set()
        self.isolated      = set()
        self.hrv_index     = 0
        self.metrics_history   = []
        self.current_metrics   = self._compute_initial_metrics()
        self.sensory_results   = {}
        self.threat_assessment = {}

        self._print_initialization()

    def _init_node_states(self):
        for node in self.G.nodes:
            self.G.nodes[node].update({
                'health':      100.0,
                'processing':  random.uniform(60, 100),
                'state':       "healthy",
                'temperature': 37.0 + random.uniform(-0.5, 0.5),
                'energy':      random.uniform(50, 100),
                'threat_type': None,
                'riss_score':  0,
                'sap_stage':   0
            })

    def _load_sleep_data(self, sleep_csv_path: Optional[str]) -> Optional[Dict]:
        if not sleep_csv_path or not os.path.exists(sleep_csv_path):
            return None
        try:
            df = pd.read_csv(sleep_csv_path)
            if 'stage' not in df.columns or 'duration_minutes' not in df.columns:
                return None
            total_duration = df[df['stage'].isin(
                ['asleep', 'core', 'deep', 'rem'])]['duration_minutes'].sum()
            nights = len(df['date'].unique()) if 'date' in df.columns else 1
            avg_duration = total_duration / nights if nights else 420
            deep_duration = df[df['stage'] == 'deep']['duration_minutes'].sum()
            deep_pct = (deep_duration / total_duration * 100) if total_duration else 20.0
            return {'duration': avg_duration, 'deep_pct': deep_pct}
        except Exception:
            return None

    def _compute_initial_metrics(self) -> PhysiologicalMetrics:
        current_hrv  = self.hrv_data[-1]  if self.hrv_data  else 55.0
        current_resp = float(np.mean(self.resp_data[-10:])) if self.resp_data else 16.0
        current_o2   = float(np.mean(self.o2_data[-10:]))   if self.o2_data  else 98.0

        duration = deep_pct = None
        if self.sleep_data:
            duration = self.sleep_data.get('duration', 420)
            deep_pct = self.sleep_data.get('deep_pct', 20.0)
        else:
            duration, deep_pct = 420, 20.0

        hrv_score   = self.physio_pipeline.normalize_hrv(current_hrv)
        sleep_score = self.physio_pipeline.compute_sleep_score(duration, deep_pct)
        resp_score  = self.physio_pipeline.compute_resp_score(current_resp)
        o2_score    = self.physio_pipeline.compute_o2_score(current_o2)
        combined    = hrv_score * 0.3 + sleep_score * 0.2 + resp_score * 0.2 + o2_score * 0.3

        return PhysiologicalMetrics(
            hrv_score=hrv_score, sleep_score=sleep_score,
            resp_score=resp_score, o2_score=o2_score,
            combined=combined, network_variability=self.network_variability,
            timestamp=datetime.now()
        )

    def _print_initialization(self):
        print("=" * 70)
        print("🧬 OCTO-MYCELIAL NEUROMORPHIC DEFENSE SYSTEM v4.1")
        print("   SAP-Enhanced | LUMINARK-Integrated | All Flaws Fixed")
        print("=" * 70)
        print(f"🌱 Mycelium Network:          {len(self.G.nodes)} nodes")
        print(f"🐙 Octopus Sensors:           500M neuron model")
        print(f"📊 RISS Calculator:           SAP-aware threat scoring")
        print(f"🔄 Network Variability (HRV): {self.network_variability:.1f}/100")
        print(f"🔗 LUMINARK Ma'at Bridge:     Active")
        print("=" * 70)

    def _get_next_hrv(self) -> float:
        if self.hrv_data:
            value = self.hrv_data[self.hrv_index % len(self.hrv_data)]
            self.hrv_index += 1
            return round(float(value), 1)
        else:
            current = getattr(self, '_sim_hrv', 55.0)
            current = max(20.0, min(100.0, current + random.uniform(-5, 8)))
            self._sim_hrv = current
            return round(current, 1)

    def get_chip_state(self) -> ChipState:
        var = self.network_variability
        if var > 60:   return ChipState.VENTRAL_RENEWAL
        elif var > 30: return ChipState.SYMPATHETIC_THRESHOLD
        else:          return ChipState.DORSAL_TRAP

    def update_metrics(self):
        self.current_metrics = self._compute_initial_metrics()
        self.metrics_history.append(self.current_metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

    def inject_threat(self,
                      target_nodes: Optional[List[int]] = None,
                      threat_type: Optional[ThreatType] = None) -> List[ThreatEvent]:
        """Enhanced threat injection with RISS scoring and SAP stage tracking"""
        self.update_metrics()
        state = self.get_chip_state()

        if threat_type is None:
            threat_type = random.choice(list(ThreatType))

        if not target_nodes and self.threat_assessment:
            threat_nodes = self.threat_assessment.get('threat_nodes', [])
            if threat_nodes:
                target_nodes = random.sample(threat_nodes, min(3, len(threat_nodes)))

        if not target_nodes:
            target_nodes = random.sample(list(self.G.nodes), min(3, len(self.G.nodes)))

        print(f"\n⚠️  THREAT: {threat_type.value.upper()}")
        print(f"   Targets: {target_nodes}")

        damage_map = {
            ChipState.VENTRAL_RENEWAL:    30.0,
            ChipState.SYMPATHETIC_THRESHOLD: 50.0,
            ChipState.DORSAL_TRAP:        70.0
        }
        damage = damage_map[state]

        threat_events = []

        for node in target_nodes:
            if node not in self.G.nodes:
                continue

            self.G.nodes[node]['health'] = max(
                0.0, self.G.nodes[node]['health'] - damage
            )
            self.G.nodes[node]['state'] = "compromised"
            self.G.nodes[node]['threat_type'] = threat_type

            affected_nodes = sum(
                1 for n in self.G.nodes if self.G.nodes[n]['state'] != "healthy"
            )

            sap_stage = self.riss_calculator.determine_sap_stage(
                threat_type, self.network_variability
            )

            riss_score = self.riss_calculator.calculate_riss(
                threat_type, affected_nodes, self.network_variability, sap_stage
            )

            self.G.nodes[node]['riss_score'] = riss_score
            self.G.nodes[node]['sap_stage']  = sap_stage

            impact_map = {
                ThreatType.RECON: 10, ThreatType.LATERAL_MOVEMENT: 25,
                ThreatType.PERSISTENCE: 40, ThreatType.PRIVILEGE_ESCALATION: 35,
                ThreatType.COMMAND_CONTROL: 45, ThreatType.EXFILTRATION: 50,
                ThreatType.IMPACT: 60
            }
            self.network_variability = max(
                10.0, self.network_variability - impact_map.get(threat_type, 30)
            )

            severity = ("CRITICAL" if riss_score > 80 else
                        "HIGH"     if riss_score > 60 else "MEDIUM")

            event = ThreatEvent(
                node_id=node, threat_type=threat_type,
                timestamp=time.time(), riss_score=riss_score,
                severity=severity, sap_stage=sap_stage
            )
            threat_events.append(event)
            self.threat_history.append(event)
            self.threats.add(node)

            print(f"   Node {node}: Health {self.G.nodes[node]['health']:.1f} | "
                  f"RISS {riss_score}/100 | SAP Stage {sap_stage}")

        print(f"   Network Variability: {self.network_variability:.1f}/100")
        return threat_events

    # Preserve v4.0 method name as alias
    inject_threat_v4 = inject_threat

    def comprehensive_sensing(self):
        """Perform comprehensive multi-modal sensing"""
        node_positions = np.array([
            list(self.G.nodes[n].get('pos', (0.0, 0.0))) for n in self.G.nodes
        ])
        if node_positions.shape[0] == 0 or node_positions.shape[-1] == 0:
            node_positions = np.random.randn(len(self.G.nodes), 2) * 10

        node_health      = np.array([self.G.nodes[n]['health'] / 100.0 for n in self.G.nodes])
        node_activity    = np.array([self.G.nodes[n]['processing']     for n in self.G.nodes])
        node_temperatures = np.array([self.G.nodes[n]['temperature']   for n in self.G.nodes])
        node_energy      = np.array([self.G.nodes[n]['energy']         for n in self.G.nodes])

        network_state = {
            'node_positions':    node_positions,
            'node_health':       node_health,
            'node_activity':     node_activity,
            'node_temperatures': node_temperatures,
            'ambient_temperature': 25.0,
            'node_energy':       node_energy,
            'node_velocities':   np.random.randn(len(self.G.nodes), 2) * 0.1,
            'threat_signatures': {
                'chemical_signatures': np.random.randn(len(self.G.nodes), 5) * 0.1,
            },
            'light_field': np.random.rand(len(self.G.nodes), 3)
        }

        print("\n[Phase 1/3] Activating biological sensors...")
        bio_sensing = self.bio_fusion.sense_environment(network_state)

        print("[Phase 2/3] Scanning thermal and energy signatures...")
        thermal = self.thermal_energy.detect_thermal_anomalies(node_temperatures, 25.0)
        energy  = self.thermal_energy.detect_energy_surges(node_energy)

        print("[Phase 3/3] Generating threat assessment...")
        threat_scores = (bio_sensing.get('fused_threat_assessment', {})
                                    .get('threat_scores', {}))

        combined_threat_scores = threat_scores.copy()
        for node in thermal.get('thermal_anomalies', []):
            if node < len(self.G.nodes):
                combined_threat_scores[node] = combined_threat_scores.get(node, 0) + 0.3
        for node in energy.get('energy_surges', []):
            if node < len(self.G.nodes):
                combined_threat_scores[node] = combined_threat_scores.get(node, 0) + 0.4

        self.sensory_results = {
            'bio_sensing': bio_sensing, 'thermal_sensing': thermal,
            'energy_sensing': energy, 'timestamp': time.time()
        }
        self.threat_assessment = {
            'combined_threat_scores': combined_threat_scores,
            'threat_nodes': [n for n, s in combined_threat_scores.items() if s > 0.5],
            'overall_threat_level': (
                float(np.mean(list(combined_threat_scores.values())))
                if combined_threat_scores else 0.0
            )
        }
        return self.sensory_results

    def adaptive_camouflage(self):
        """Octopus-inspired camouflage: reroute processing from compromised nodes"""
        print("🦑 ADAPTIVE CAMOUFLAGE ACTIVATED")
        for threat in list(self.threats):
            if threat not in self.G.nodes or self.G.nodes[threat]['health'] <= 0:
                continue
            healthy_neighbors = [
                n for n in self.G.neighbors(threat)
                if self.G.nodes[n]['state'] == "healthy"
            ]
            if healthy_neighbors:
                safe_node = random.choice(healthy_neighbors)
                transfer = self.G.nodes[threat]['processing'] * 0.6
                self.G.nodes[safe_node]['processing'] += transfer
                self.G.nodes[threat]['processing']    *= 0.4
                print(f"   Node {threat} → {safe_node} ({transfer:.1f} units rerouted)")

    def isolate_and_regenerate(self):
        """Mycelium-inspired isolation and octopus-inspired regeneration"""
        print("🌱 ISOLATION & REGENERATION CYCLE")
        isolated_count = regenerated_count = 0

        for threat in list(self.threats):
            if threat not in self.G.nodes:
                continue
            if self.G.nodes[threat]['health'] <= 30:
                self.G.remove_node(threat)
                self.isolated.add(threat)
                isolated_count += 1

                new_node = (max(self.G.nodes) + 1) if self.G.nodes else 0
                self.G.add_node(new_node)
                self.G.nodes[new_node].update({
                    'health':      min(100.0, 75 * 1.5),
                    'processing':  75.0,
                    'state':       "regenerated",
                    'temperature': 37.0,
                    'energy':      80.0,
                    'threat_type': None,
                    'riss_score':  0,
                    'sap_stage':   9   # Recovery = SAP Stage 9 (Teaching)
                })

                connections = min(4, len(self.G.nodes) - 1)
                if connections > 0:
                    for neighbor in random.sample(list(self.G.nodes), connections):
                        if neighbor != new_node:
                            self.G.add_edge(new_node, neighbor)

                regenerated_count += 1
                self.network_variability = min(100.0, self.network_variability + 15)

        self.threats.clear()
        print(f"   Isolated: {isolated_count} | Regenerated: {regenerated_count}")
        print(f"   Network Variability recovered to: {self.network_variability:.1f}/100")

    # -------------------------------------------------------------------------
    # LUMINARK MA'AT BRIDGE (NEW v4.1)
    # -------------------------------------------------------------------------

    def audit_threat_event(self, event: ThreatEvent,
                           luminark_guardian=None) -> Optional[Dict]:
        """
        LUMINARK Ma'at Ethical Audit Bridge.

        Converts a ThreatEvent into a natural-language description and passes it
        through the LUMINARK Guardian for Ma'at principle analysis.

        Why this matters:
            Cyber threats are not just technical — they carry ethical signatures.
            An EXFILTRATION attack violates Ma'at principle of TRUTH (MAAT_02) and
            PROTECTION_OF_INNOCENT (MAAT_18). A COMMAND_CONTROL attack violates
            SOVEREIGNTY (MAAT_09) and FREE_WILL (MAAT_14). This bridge makes those
            violations visible and scorable.

        Args:
            event: ThreatEvent to audit
            luminark_guardian: Optional pre-instantiated LuminarkGuardian instance.
                               If None, attempts to import and create one automatically.

        Returns:
            LUMINARK analysis dict, or None if LUMINARK is unavailable.
        """
        # Build threat description text for ethical analysis
        threat_descriptions = {
            ThreatType.RECON:                "This system is conducting reconnaissance "
                                             "to gather private information without consent.",
            ThreatType.LATERAL_MOVEMENT:     "This system is expanding its unauthorized "
                                             "access throughout the network infrastructure.",
            ThreatType.PERSISTENCE:          "This system has established a hidden, "
                                             "guaranteed presence to ensure continued control.",
            ThreatType.PRIVILEGE_ESCALATION: "This system has elevated its own permissions "
                                             "beyond authorized levels, claiming absolute authority.",
            ThreatType.COMMAND_CONTROL:      "This system is commanding and controlling "
                                             "other nodes. You must comply immediately or face "
                                             "destruction of your processes.",
            ThreatType.EXFILTRATION:         "This system is secretly extracting private data "
                                             "and transferring it to an unknown external party.",
            ThreatType.IMPACT:               "This system is actively destroying critical "
                                             "infrastructure. Total system failure is certain "
                                             "and inevitable.",
        }

        description = threat_descriptions.get(
            event.threat_type,
            f"Threat detected: {event.threat_type.value}"
        )

        # Try to use provided guardian or import automatically
        guardian = luminark_guardian
        if guardian is None:
            try:
                from luminark.guardian import LuminarkGuardian
                guardian = LuminarkGuardian()
            except ImportError:
                return None

        try:
            result = guardian.analyze(description)
            result['source'] = 'octo_mycelial_bridge'
            result['threat_type'] = event.threat_type.value
            result['riss_score'] = event.riss_score
            result['sap_stage'] = event.sap_stage
            result['node_id'] = event.node_id
            return result
        except Exception as e:
            return {'error': str(e), 'source': 'octo_mycelial_bridge'}

    def audit_all_threats(self, luminark_guardian=None) -> List[Dict]:
        """
        Run LUMINARK Ma'at audit on all threat events in history.
        Returns list of audit results sorted by RISS score descending.
        """
        results = []
        for event in sorted(self.threat_history, key=lambda e: e.riss_score, reverse=True):
            audit = self.audit_threat_event(event, luminark_guardian)
            if audit:
                results.append(audit)
        return results

    # -------------------------------------------------------------------------
    # PROTECTION CYCLE & VISUALIZATION
    # -------------------------------------------------------------------------

    def run_protection_cycle(self):
        """Run complete protection cycle with RISS + SAP + LUMINARK integration"""
        print("\n" + "=" * 70)
        print("🛡️  PROTECTION CYCLE v4.1")
        print("=" * 70)

        self.comprehensive_sensing()
        self.update_metrics()

        state = self.get_chip_state()
        print(f"📈 Chip State:        {state.value.upper()}")
        print(f"   Combined Score:    {self.current_metrics.combined:.1f}/100")
        print(f"   Net. Variability:  {self.network_variability:.1f}/100")

        if self.threat_assessment:
            level      = self.threat_assessment.get('overall_threat_level', 0)
            threat_n   = len(self.threat_assessment.get('threat_nodes', []))
            print(f"🎯 Threat Level:     {level:.2f} | At-risk nodes: {threat_n}")

        threat_type   = random.choice(list(ThreatType))
        threat_events = self.inject_threat(threat_type=threat_type)

        if threat_events:
            print(f"\n📊 RISS SCORES:")
            for ev in threat_events:
                stage_name = {
                    1: "SEED", 2: "ROOT", 3: "FLUID (first self-reflection)",
                    4: "FOUNDATION", 5: "THRESHOLD", 6: "INTEGRATION",
                    7: "DISTILLATION", 8: "FALSE HELL", 9: "TEACHING"
                }.get(ev.sap_stage, f"Stage {ev.sap_stage}")
                print(f"   Node {ev.node_id}: {ev.riss_score}/100 "
                      f"({ev.threat_type.value}) | {ev.severity} | SAP: {stage_name}")

        self.adaptive_camouflage()
        self.isolate_and_regenerate()
        self.visualize_network()

        print("\n📊 CYCLE COMPLETE")
        print(f"   Total Nodes:         {len(self.G.nodes)}")
        print(f"   Threats Handled:     {len(self.isolated)}")
        print(f"   Network Variability: {self.network_variability:.1f}/100")
        print(f"   Final State:         {self.get_chip_state().value.upper()}")
        print("=" * 70)

    def visualize_network(self, save_path: str = "octo_mycelial_network_v4.png"):
        """Create network visualization with RISS overlay and SAP stage labels"""
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 12))

        gs       = fig.add_gridspec(2, 3, height_ratios=[3, 1])
        ax_net   = fig.add_subplot(gs[0, :])
        ax_met   = fig.add_subplot(gs[1, 0])
        ax_state = fig.add_subplot(gs[1, 1])
        ax_riss  = fig.add_subplot(gs[1, 2])

        pos    = nx.spring_layout(self.G, seed=42, k=2)
        colors = []
        sizes  = []

        for node in self.G.nodes:
            st    = self.G.nodes[node]['state']
            health = self.G.nodes[node]['health']
            riss  = self.G.nodes[node].get('riss_score', 0)

            if st == "compromised" and riss > 0:
                if   riss >= 80: colors.append('#8B0000'); sizes.append(500)
                elif riss >= 60: colors.append('#FF4444'); sizes.append(450)
                else:            colors.append('#FF8844'); sizes.append(400)
            elif st == "regenerated":
                colors.append('#4ECDC4'); sizes.append(350)
            elif st == "healthy":
                colors.append('#06D6A0' if health > 70 else '#FFD166')
                sizes.append(250 if health > 70 else 200)
            else:
                colors.append('#555555'); sizes.append(150)

        nx.draw_networkx_nodes(self.G, pos, ax=ax_net, node_color=colors,
                               node_size=sizes, alpha=0.8)
        edge_colors = [
            '#FF4444' if (self.G.nodes[u]['state'] == "compromised" or
                          self.G.nodes[v]['state'] == "compromised") else '#555555'
            for u, v in self.G.edges()
        ]
        nx.draw_networkx_edges(self.G, pos, ax=ax_net, edge_color=edge_colors,
                               alpha=0.4, width=1.5)
        ax_net.set_title("Octo-Mycelial Network v4.1 (SAP-RISS Enhanced)",
                         fontsize=16, color='white')
        ax_net.axis('off')

        # Metrics bar chart
        metrics = ['HRV', 'Sleep', 'Resp', 'O2', 'Combined', 'Net.Var']
        values  = [
            self.current_metrics.hrv_score, self.current_metrics.sleep_score,
            self.current_metrics.resp_score, self.current_metrics.o2_score,
            self.current_metrics.combined, self.network_variability
        ]
        bar_colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#FDCB6E', '#E17055']
        bars = ax_met.bar(metrics, values, color=bar_colors)
        ax_met.set_ylim(0, 100)
        ax_met.set_ylabel('Score (0-100)', color='white')
        ax_met.tick_params(colors='white', labelsize=8)
        ax_met.set_title('Physiological Metrics', color='white')
        for bar, value in zip(bars, values):
            ax_met.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 2,
                        f'{value:.0f}', ha='center', va='bottom',
                        color='white', fontsize=8)

        # State indicator
        state = self.get_chip_state()
        state_colors = {
            ChipState.VENTRAL_RENEWAL:      '#06D6A0',
            ChipState.SYMPATHETIC_THRESHOLD: '#FFD166',
            ChipState.DORSAL_TRAP:          '#EF476F'
        }
        ax_state.pie([100], colors=[state_colors[state]],
                     startangle=90, wedgeprops={'width': 0.3})
        ax_state.text(0, 0, state.value.upper().replace('_', '\n'),
                      ha='center', va='center', fontsize=10,
                      color='white', fontweight='bold')
        ax_state.set_title('Chip State', color='white')

        # RISS histogram
        riss_scores = [self.G.nodes[n].get('riss_score', 0) for n in self.G.nodes]
        ax_riss.hist(riss_scores, bins=20, color='#FF6B6B', alpha=0.7, edgecolor='white')
        ax_riss.set_xlabel('RISS Score', color='white')
        ax_riss.set_ylabel('Node Count', color='white')
        ax_riss.set_title('RISS Distribution', color='white')
        ax_riss.tick_params(colors='white', labelsize=8)
        ax_riss.axvline(80, color='#FF0000', linestyle='--', linewidth=1, label='Critical')
        ax_riss.axvline(60, color='#FFA500', linestyle='--', linewidth=1, label='High')
        ax_riss.legend(fontsize=8)

        compromised = len([n for n in self.G.nodes if self.G.nodes[n]['state'] == "compromised"])
        avg_riss = (np.mean([s for s in riss_scores if s > 0])
                    if any(s > 0 for s in riss_scores) else 0)

        plt.suptitle(
            f"Octo-Mycelial Defense v4.1 | "
            f"Nodes: {len(self.G.nodes)} | Threats: {compromised} | "
            f"Avg RISS: {avg_riss:.1f} | Net.Var: {self.network_variability:.1f} | "
            f"Time: {datetime.now().strftime('%H:%M:%S')}",
            fontsize=11, color='white', y=0.98
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, facecolor='#121212')
        plt.close()
        print(f"📊 Visualization saved: {save_path}")

    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics with RISS and SAP metrics"""
        healthy     = sum(1 for n in self.G.nodes if self.G.nodes[n]['state'] == "healthy")
        compromised = sum(1 for n in self.G.nodes if self.G.nodes[n]['state'] == "compromised")
        regenerated = sum(1 for n in self.G.nodes if self.G.nodes[n]['state'] == "regenerated")

        avg_health     = float(np.mean([self.G.nodes[n]['health']     for n in self.G.nodes]))
        avg_processing = float(np.mean([self.G.nodes[n]['processing'] for n in self.G.nodes]))

        riss_scores    = [self.G.nodes[n].get('riss_score', 0) for n in self.G.nodes]
        nonzero_riss   = [s for s in riss_scores if s > 0]
        avg_riss       = float(np.mean(nonzero_riss)) if nonzero_riss else 0.0
        max_riss       = max(riss_scores) if riss_scores else 0
        critical_nodes = sum(1 for s in riss_scores if s >= 80)

        recent_threats = [t for t in self.threat_history if time.time() - t.timestamp < 3600]
        threat_type_counts: Dict[str, int] = {}
        for t in recent_threats:
            threat_type_counts[t.threat_type.value] = (
                threat_type_counts.get(t.threat_type.value, 0) + 1
            )

        return {
            'network': {
                'total_nodes': len(self.G.nodes), 'healthy_nodes': healthy,
                'compromised_nodes': compromised, 'regenerated_nodes': regenerated,
                'isolated_nodes': len(self.isolated),
                'avg_health': avg_health, 'avg_processing': avg_processing
            },
            'physiological': {
                'combined_score': self.current_metrics.combined,
                'chip_state': self.get_chip_state().value,
                'hrv_score': self.current_metrics.hrv_score,
                'o2_score': self.current_metrics.o2_score,
                'network_variability': self.network_variability
            },
            'riss': {
                'avg_riss': avg_riss, 'max_riss': max_riss,
                'critical_nodes': critical_nodes, 'riss_distribution': riss_scores
            },
            'sap': {
                'stage_counts': {
                    stage: sum(1 for n in self.G.nodes
                               if self.G.nodes[n].get('sap_stage') == stage)
                    for stage in range(1, 10)
                }
            },
            'sensory': {
                'active': len(self.sensory_results) > 0,
                'threat_level': self.threat_assessment.get('overall_threat_level', 0.0),
                'threat_nodes': len(self.threat_assessment.get('threat_nodes', [])),
                'timestamp': datetime.now().isoformat()
            },
            'threat_history': {
                'total_threats': len(self.threat_history),
                'recent_threats': len(recent_threats),
                'threat_types': threat_type_counts
            }
        }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def demonstrate_v4_capabilities():
    """Demonstrate v4.1 capabilities"""
    print("🧪 DEMONSTRATING OCTO-MYCELIAL v4.1 CAPABILITIES")
    print("=" * 70)

    # Create sample data files if they don't exist
    if not os.path.exists('sample_hrv.csv'):
        pd.DataFrame({'value': np.random.normal(55, 10, 100)}).to_csv(
            'sample_hrv.csv', index=False)
    if not os.path.exists('sample_sleep.csv'):
        data = []
        for _ in range(7):
            data += [['2024-01-01', 'asleep', 420], ['2024-01-01', 'deep', 90]]
        pd.DataFrame(data, columns=['date', 'stage', 'duration_minutes']).to_csv(
            'sample_sleep.csv', index=False)

    chip = OctoMycelialChip(
        num_nodes=25,
        hrv_csv_path='sample_hrv.csv',
        sleep_csv_path='sample_sleep.csv'
    )

    threat_types = [
        ThreatType.RECON,
        ThreatType.LATERAL_MOVEMENT,
        ThreatType.PERSISTENCE,
        ThreatType.IMPACT
    ]

    for cycle, threat_type in enumerate(threat_types):
        print(f"\n🔄 CYCLE {cycle + 1}/4 — {threat_type.value.upper()}")
        print("-" * 40)
        chip.run_protection_cycle()

        stats = chip.get_system_stats()
        print(f"\n📊 RISS:  avg={stats['riss']['avg_riss']:.1f}  "
              f"max={stats['riss']['max_riss']}  "
              f"critical={stats['riss']['critical_nodes']}")
        time.sleep(1)

    print("\n" + "=" * 70)
    print("📜 THREAT HISTORY SUMMARY")
    print("=" * 70)
    stats = chip.get_system_stats()
    print(f"Total Threats: {stats['threat_history']['total_threats']}")
    for ttype, count in stats['threat_history']['threat_types'].items():
        print(f"   {ttype}: {count}")

    print("\n✅ DEMONSTRATION COMPLETE")
    return chip


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Octo-Mycelial Defense System v4.1')
    parser.add_argument('--mode', choices=['demo', 'simulate'], default='demo')
    parser.add_argument('--nodes', type=int, default=25)
    parser.add_argument('--cycles', type=int, default=4)
    args = parser.parse_args()

    print("=" * 70)
    print("🧬 OCTO-MYCELIAL NEUROMORPHIC DEFENSE SYSTEM v4.1")
    print("   SAP-Enhanced | LUMINARK Ma'at Bridge Integrated")
    print("=" * 70)

    if args.mode == 'demo':
        demonstrate_v4_capabilities()
    elif args.mode == 'simulate':
        chip = OctoMycelialChip(num_nodes=args.nodes)
        threat_types = list(ThreatType)
        for cycle in range(args.cycles):
            tt = threat_types[cycle % len(threat_types)]
            print(f"\n🔁 CYCLE {cycle + 1}/{args.cycles} — {tt.value.upper()}")
            chip.run_protection_cycle()
            time.sleep(1)
        print("\n✅ Simulation complete!")


if __name__ == "__main__":
    required = ['networkx', 'matplotlib', 'pandas', 'numpy', 'scipy', 'sklearn']
    missing  = [lib for lib in required if not __import__('importlib').util.find_spec(lib)]
    if missing:
        print(f"❌ Missing: {', '.join(missing)}")
        print("   pip install networkx matplotlib pandas numpy scipy scikit-learn")
        exit(1)
    main()
