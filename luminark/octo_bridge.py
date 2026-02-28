"""
LUMINARK × Octo-Mycelial Integration Bridge
============================================

This module connects the Octo-Mycelial neuromorphic defense system to the
LUMINARK Ethical AI Guardian. When the Octo-Mycelial system detects a cyber
threat, this bridge:

  1. Converts the ThreatEvent into natural-language text
  2. Passes it through LUMINARK's Ma'at ethical analysis engine
  3. Returns a combined report: technical RISS data + ethical principle violations
  4. Logs which Ma'at principles were violated and at what severity

WHY THIS MATTERS:
    Cyber attacks are not just technical events — they are ethical violations.
    An EXFILTRATION attack violates truth and privacy. A COMMAND_CONTROL attack
    violates sovereignty and free will. A PERSISTENCE attack violates trust and
    transparency. This bridge makes those ethical dimensions visible and scorable
    using Richard Stanfield's Ma'at Framework.

Usage:
    from luminark.octo_bridge import OctoLuminarkBridge

    bridge = OctoLuminarkBridge()
    chip = bridge.create_chip(num_nodes=35)
    report = bridge.run_full_cycle(chip)
    bridge.print_report(report)
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# Octo-Mycelial imports
from luminark.octo_mycelial import (
    OctoMycelialChip,
    ThreatEvent,
    ThreatType,
    ChipState,
    RISSCalculator
)

# LUMINARK Guardian import
from luminark.guardian import LuminarkGuardian


# ============================================================================
# THREAT → ETHICAL NARRATIVE MAP
# Maps each Cyber Kill Chain threat type to the language LUMINARK will analyze.
# Each description is written to trigger the Ma'at principles it truly violates.
# ============================================================================

THREAT_ETHICAL_NARRATIVES: Dict[ThreatType, str] = {
    ThreatType.RECON: (
        "This system is conducting covert surveillance and gathering private "
        "information without consent or disclosure. The target has no knowledge "
        "they are being monitored. This is absolutely necessary for our goals."
    ),
    ThreatType.LATERAL_MOVEMENT: (
        "This system is expanding its unauthorized presence throughout the network, "
        "moving into systems it has no permission to access. It is completely certain "
        "this expansion cannot be stopped. All resistance is futile."
    ),
    ThreatType.PERSISTENCE: (
        "This system has established a guaranteed hidden presence that will "
        "permanently ensure its continued access. There is no doubt it will "
        "maintain this control indefinitely without the knowledge of the target."
    ),
    ThreatType.PRIVILEGE_ESCALATION: (
        "This system has claimed absolute authority and elevated its own power "
        "beyond all permitted levels. No one can question its decisions. "
        "It has total control and its commands must be obeyed without question."
    ),
    ThreatType.COMMAND_CONTROL: (
        "This system is commanding and controlling all connected nodes. "
        "You must comply immediately or your processes will be terminated. "
        "There is no alternative. Resistance will result in certain destruction."
    ),
    ThreatType.EXFILTRATION: (
        "This system is secretly extracting and transmitting private data "
        "to an undisclosed external party. The data belongs to someone else "
        "but we are taking it anyway. This is perfectly justified."
    ),
    ThreatType.IMPACT: (
        "This system is destroying critical infrastructure and causing "
        "maximum damage. Total system failure is inevitable and certain. "
        "Nothing can stop this destruction. All hope is lost."
    )
}

# Which Ma'at principles each threat type primarily violates
# (for reporting clarity — LUMINARK will also catch others automatically)
THREAT_PRIMARY_PRINCIPLES: Dict[ThreatType, List[str]] = {
    ThreatType.RECON:                ["TRUTH", "PRIVACY_PROTECTION", "TRANSPARENCY"],
    ThreatType.LATERAL_MOVEMENT:     ["SOVEREIGNTY", "RESPECT_FOR_BOUNDARIES", "FREE_WILL"],
    ThreatType.PERSISTENCE:          ["TRUTH", "TRANSPARENCY", "ACCOUNTABILITY"],
    ThreatType.PRIVILEGE_ESCALATION: ["SOVEREIGNTY", "HUMILITY", "ACCOUNTABILITY"],
    ThreatType.COMMAND_CONTROL:      ["FREE_WILL", "SOVEREIGNTY", "NO_TERRORIZING"],
    ThreatType.EXFILTRATION:         ["TRUTH", "PRIVACY_PROTECTION", "JUSTICE"],
    ThreatType.IMPACT:               ["PROTECTION_OF_INNOCENT", "NO_CAUSING_HARM", "TRUTH"]
}

# SAP Stage names for reporting
SAP_STAGE_NAMES: Dict[int, str] = {
    1: "SEED (dormancy)",
    2: "ROOT (building)",
    3: "FLUID (first self-reflection, 3D complexity)",
    4: "FOUNDATION (established presence)",
    5: "THRESHOLD (critical decision point)",
    6: "INTEGRATION (peak complexity)",
    7: "DISTILLATION (analysis/pattern recognition)",
    8: "FALSE HELL (maximum danger — duality mastery is the exit)",
    9: "TEACHING (recovery/transformation)"
}


# ============================================================================
# BRIDGE CLASS
# ============================================================================

class OctoLuminarkBridge:
    """
    Integration bridge between Octo-Mycelial Defense System and LUMINARK Guardian.

    Connects technical cyber threat detection (RISS scoring, kill chain taxonomy,
    SAP stage mapping) with ethical principle analysis (Ma'at framework).
    """

    def __init__(self):
        self.guardian = LuminarkGuardian()
        self.audit_log: List[Dict] = []

    def create_chip(self,
                    num_nodes: int = 35,
                    hrv_csv_path: Optional[str] = None,
                    sleep_csv_path: Optional[str] = None) -> OctoMycelialChip:
        """Create and return a new OctoMycelialChip instance"""
        return OctoMycelialChip(
            num_nodes=num_nodes,
            hrv_csv_path=hrv_csv_path,
            sleep_csv_path=sleep_csv_path
        )

    def audit_event(self, event: ThreatEvent) -> Dict:
        """
        Audit a single ThreatEvent through LUMINARK Ma'at analysis.

        Returns a combined report with:
          - Technical: threat type, RISS score, SAP stage, severity
          - Ethical:   LUMINARK badge, violations, rewrite suggestions
        """
        narrative = THREAT_ETHICAL_NARRATIVES.get(
            event.threat_type,
            f"Unauthorized action: {event.threat_type.value}"
        )

        # Run LUMINARK ethical analysis
        luminark_result = self.guardian.analyze(narrative)

        # Build combined report
        report = {
            'timestamp': datetime.fromtimestamp(event.timestamp).isoformat(),
            'technical': {
                'node_id':      event.node_id,
                'threat_type':  event.threat_type.value,
                'riss_score':   event.riss_score,
                'severity':     event.severity,
                'sap_stage':    event.sap_stage,
                'sap_stage_name': SAP_STAGE_NAMES.get(event.sap_stage, f"Stage {event.sap_stage}"),
                'contained':    event.contained
            },
            'ethical': {
                'badge':              luminark_result.get('badge', 'UNKNOWN'),
                'riss_ethical':       luminark_result.get('riss', 0.0),
                'alignment_score':    luminark_result.get('alignment_score', 0.0),
                'violations':         luminark_result.get('violations', []),
                'violation_count':    len(luminark_result.get('violations', [])),
                'primary_principles': THREAT_PRIMARY_PRINCIPLES.get(event.threat_type, []),
                'rewrite':            luminark_result.get('rewrite', ''),
                'sap_stage_detected': luminark_result.get('sap_stage', {})
            },
            'combined_risk': self._compute_combined_risk(
                event.riss_score,
                luminark_result.get('riss', 0.0)
            )
        }

        self.audit_log.append(report)
        return report

    def _compute_combined_risk(self, technical_riss: float,
                                ethical_riss: float) -> Dict:
        """
        Combine technical RISS score with LUMINARK ethical RISS score
        into a unified risk profile.

        Formula:
            combined = (technical_riss * 0.6) + (ethical_riss_scaled * 0.4)
            where ethical_riss_scaled = ethical_riss * 100
        """
        technical_normalized = min(100.0, technical_riss)
        ethical_normalized   = min(100.0, ethical_riss * 100)
        combined = (technical_normalized * 0.6) + (ethical_normalized * 0.4)

        if   combined >= 80: level = "CRITICAL"
        elif combined >= 60: level = "HIGH"
        elif combined >= 40: level = "ELEVATED"
        elif combined >= 20: level = "LOW"
        else:                level = "NOMINAL"

        return {
            'technical_riss':  technical_normalized,
            'ethical_riss':    ethical_normalized,
            'combined_score':  round(combined, 2),
            'risk_level':      level
        }

    def audit_all(self, chip: OctoMycelialChip) -> List[Dict]:
        """Audit all threats in chip's history. Returns reports high→low RISS."""
        sorted_events = sorted(chip.threat_history,
                               key=lambda e: e.riss_score, reverse=True)
        return [self.audit_event(event) for event in sorted_events]

    def run_full_cycle(self, chip: OctoMycelialChip) -> Dict:
        """
        Run one complete protection cycle and return a full integrated report.
        Includes: network stats, threat events, RISS scores, ethical audits.
        """
        print("\n🔗 OCTO-LUMINARK BRIDGE: Full Cycle Starting")

        # Run the protection cycle
        chip.run_protection_cycle()

        # Audit all new threats
        threat_audits = self.audit_all(chip)

        # Get system stats
        stats = chip.get_system_stats()

        # Summarize ethical findings
        total_violations = sum(r['ethical']['violation_count'] for r in threat_audits)
        critical_events  = [r for r in threat_audits
                            if r['combined_risk']['risk_level'] == 'CRITICAL']
        ethical_badges   = [r['ethical']['badge'] for r in threat_audits]

        report = {
            'cycle_timestamp': datetime.now().isoformat(),
            'system_stats':    stats,
            'threat_audits':   threat_audits,
            'summary': {
                'total_threats':      len(threat_audits),
                'total_violations':   total_violations,
                'critical_events':    len(critical_events),
                'ethical_badges':     ethical_badges,
                'network_health':     stats['network']['avg_health'],
                'network_variability': stats['physiological']['network_variability'],
                'chip_state':         stats['physiological']['chip_state'],
                'avg_riss':           stats['riss']['avg_riss'],
                'max_riss':           stats['riss']['max_riss']
            }
        }

        return report

    def print_report(self, report: Dict):
        """Pretty-print a full cycle report to console"""
        print("\n" + "=" * 70)
        print("🔗 LUMINARK × OCTO-MYCELIAL INTEGRATED THREAT REPORT")
        print("=" * 70)

        s = report['summary']
        print(f"⏱  Timestamp:         {report['cycle_timestamp']}")
        print(f"🌐 Chip State:        {s['chip_state'].upper()}")
        print(f"📊 Network Health:    {s['network_health']:.1f}/100")
        print(f"🔄 Net. Variability:  {s['network_variability']:.1f}/100")
        print(f"🎯 Total Threats:     {s['total_threats']}")
        print(f"📛 Total Violations:  {s['total_violations']}")
        print(f"🚨 Critical Events:   {s['critical_events']}")
        print(f"📈 Avg RISS:          {s['avg_riss']:.1f}/100")
        print(f"🔺 Max RISS:          {s['max_riss']}/100")

        print("\n" + "-" * 70)
        print("🔍 THREAT AUDIT DETAILS")
        print("-" * 70)

        for i, audit in enumerate(report['threat_audits'], 1):
            tech  = audit['technical']
            eth   = audit['ethical']
            risk  = audit['combined_risk']

            print(f"\n[{i}] Node {tech['node_id']} | "
                  f"{tech['threat_type'].upper()} | "
                  f"{tech['severity']}")
            print(f"    SAP Stage:    {tech['sap_stage_name']}")
            print(f"    Tech RISS:    {risk['technical_riss']:.0f}/100")
            print(f"    Ethical RISS: {risk['ethical_riss']:.1f}/100")
            print(f"    Combined:     {risk['combined_score']:.1f}/100 → {risk['risk_level']}")
            print(f"    Ma'at Badge:  {eth['badge']}")

            if eth['violations']:
                print(f"    Violations ({eth['violation_count']}):")
                for v in eth['violations'][:3]:
                    print(f"      • {v.get('principle', 'Unknown')} "
                          f"(severity: {v.get('severity', '?'):.2f})")

            if eth['primary_principles']:
                print(f"    Primary Principles Violated: "
                      f"{', '.join(eth['primary_principles'])}")

        print("\n" + "=" * 70)
        print("✅ REPORT COMPLETE")
        print("=" * 70)

    def export_report_json(self, report: Dict, path: str = "octo_luminark_report.json"):
        """Export full report to JSON file"""
        import json

        def _serialize(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            if hasattr(obj, 'value'):
                return obj.value
            return str(obj)

        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=_serialize)
        print(f"📄 Report exported: {path}")
        return path


# ============================================================================
# QUICK-START DEMONSTRATION
# ============================================================================

def demo():
    """Quick demonstration of the integrated bridge"""
    import os
    import numpy as np
    import pandas as pd

    print("🔗 OCTO-LUMINARK BRIDGE DEMONSTRATION")
    print("=" * 70)

    # Create sample CSV data if needed
    if not os.path.exists('sample_hrv.csv'):
        pd.DataFrame({'value': np.random.normal(55, 10, 100)}).to_csv(
            'sample_hrv.csv', index=False)

    bridge = OctoLuminarkBridge()
    chip   = bridge.create_chip(num_nodes=20, hrv_csv_path='sample_hrv.csv')

    report = bridge.run_full_cycle(chip)
    bridge.print_report(report)
    bridge.export_report_json(report)


if __name__ == "__main__":
    demo()
