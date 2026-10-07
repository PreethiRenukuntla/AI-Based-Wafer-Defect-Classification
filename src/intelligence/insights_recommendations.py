import json

ENGINEERING_RECOMMENDATION_DATABASE = {
    'Center': {
        'suggested_investigation': "Suggested engineering investigation: Photo-lithography & CMP Center Focus",
        'target_areas': [
            "Inspect photo-lithography stepper/scanner center focus offset and exposure dose.",
            "Review Chemical Mechanical Planarization (CMP) center down-force pressure profile.",
            "Inspect photoresist spin-coater chuck temperature and spin-speed uniformity.",
            "Check wafer table thermal chuck flatness and vacuum suction balance."
        ],
        'action_type': "Lithography & CMP Process Engineering Review"
    },
    'Donut': {
        'suggested_investigation': "Suggested engineering investigation: CVD/Etch Chamber Gas Distribution",
        'target_areas': [
            "Review Chemical Vapor Deposition (CVD) or Plasma Etch gas showerhead distribution.",
            "Inspect central gas injection nozzle for partial chemical clogging.",
            "Review plasma chamber RF power uniformity and ring magnet alignment.",
            "Check chamber exhaust pump suction uniformity."
        ],
        'action_type': "Chamber Plasma & Gas Flow Engineering Review"
    },
    'Edge-Ring': {
        'suggested_investigation': "Suggested engineering investigation: Wafer Bevel & Edge Bead Removal (EBR)",
        'target_areas': [
            "Inspect wafer edge clamp ring and focus ring for mechanical wear or erosion.",
            "Review Edge Bead Removal (EBR) solvent dispenser nozzle angle and flow rate.",
            "Inspect wafer bevel etch gas flow and edge purging system.",
            "Check wafer edge clamp contact pressure during wet processing."
        ],
        'action_type': "Edge & Bevel Process Engineering Review"
    },
    'Edge-Loc': {
        'suggested_investigation': "Suggested engineering investigation: Handling Robot Blade & Cassette Alignment",
        'target_areas': [
            "Inspect robotic transfer arm end-effector (blade) contact and alignment.",
            "Check wafer cassette slot clearance to prevent edge rubbing during transfer.",
            "Review Rapid Thermal Annealing (RTA) edge heat loss boundary.",
            "Check wafer edge mechanical pre-aligner sensor zero position."
        ],
        'action_type': "Automation & Thermal Border Inspection"
    },
    'Loc': {
        'suggested_investigation': "Suggested engineering investigation: Micro-Particle & Reticle Inspection",
        'target_areas': [
            "Inspect reticle/mask for localized particle contamination or localized scratch.",
            "Review cleanroom localized particle counter log at specific process station.",
            "Inspect optical stepper lens element cleanliness.",
            "Review localized spot handling tool contact."
        ],
        'action_type': "Particle Defect & Photolithography Reticle Audit"
    },
    'Scratch': {
        'suggested_investigation': "Suggested engineering investigation: Robotic Handler & CMP Pad Debris",
        'target_areas': [
            "Inspect robotic transfer arm end-effector and vacuum wand tips for abrasions.",
            "Check CMP polishing pad condition and slurry inline filter for coarse grit debris.",
            "Inspect manual wafer inspection tweezers and cassette handling tools.",
            "Check wafer transport tracks and loadlock mechanics."
        ],
        'action_type': "Mechanical Transport & Slurry Filtration Inspection"
    },
    'Random': {
        'suggested_investigation': "Suggested engineering investigation: Cleanroom HEPA & Airborne Particles",
        'target_areas': [
            "Inspect cleanroom HEPA/ULPA air filtration flow and laminar air speed.",
            "Review chemical delivery line micro-filters (DI water, liquid chemicals).",
            "Perform ambient particle scan across lithography and etch bays.",
            "Review operator cleanroom garment compliance log."
        ],
        'action_type': "Yield Environment & Air Quality Audit"
    },
    'Near-full': {
        'suggested_investigation': "Suggested engineering investigation: Major Process Excursion & Etch Stop",
        'target_areas': [
            "Trigger immediate process excursion alert for target tool bay.",
            "Review etch-stop layer selectivity and gross chemical over-etch log.",
            "Inspect main power supply stability and chemical delivery manifold shutdown.",
            "Verify raw silicon substrate wafer quality certificate."
        ],
        'action_type': "Critical Tool Line Excursion Audit"
    },
    'none': {
        'suggested_investigation': "Suggested engineering investigation: Standard Process Control Monitoring",
        'target_areas': [
            "Wafer exhibits normal die yield pattern without recognized spatial defect signature.",
            "Maintain baseline Statistical Process Control (SPC) monitoring.",
            "Proceed to standard wafer acceptance testing (WAT)."
        ],
        'action_type': "Routine Statistical Process Control (SPC)"
    }
}


def generate_ai_insights(wafer_matrix, predicted_class, confidence, yield_stats, severity_info, lot_info=None):
    """
    Generates structured AI Insights based on single wafer stats, lot context, and model predictions.
    Does NOT fabricate causal explanations.
    """
    insights = []

    # 1. Defect & Yield Insight
    if predicted_class == 'none':
        insights.append(f"Wafer classified as Normal with high yield ({yield_stats['yield_pct']}%). No significant failure pattern detected.")
    else:
        insights.append(f"Wafer exhibits a dominant spatial failure pattern classified as '{predicted_class}' with {confidence:.1f}% confidence.")
        insights.append(f"Measured die yield is {yield_stats['yield_pct']}% ({yield_stats['defective_dies']} defective dies out of {yield_stats['total_dies']} total dies).")

    # 2. Severity Insight
    insights.append(f"Model-derived analytical severity score is {severity_info['severity_score']}/100 ({severity_info['severity_rating']}).")

    # 3. Spatial Concentration Insight
    if severity_info['clustering_score'] > 0.5:
        insights.append(f"Defects show high spatial concentration (clustering score: {severity_info['clustering_score']:.2f}), suggesting a localized tool or process signature.")
    elif severity_info['clustering_score'] > 0.1:
        insights.append(f"Defects show moderate spatial grouping across the wafer surface.")

    # 4. Confidence Insight
    if confidence < 65.0:
        insights.append(f"ATTENTION: Model prediction confidence is below 65% ({confidence:.1f}%). Human-in-the-loop engineering review is recommended.")

    # 5. Lot Context Insight (if available)
    if lot_info and lot_info.get('found', False):
        insights.append(f"Lot '{lot_info['lotName']}' analysis: Average lot yield is {lot_info['avg_lot_yield']}%. Dominant lot pattern: {max(lot_info['defect_class_distribution'], key=lot_info['defect_class_distribution'].get, default='N/A')}.")

    return insights


def generate_engineering_recommendations(predicted_class, severity_info):
    """
    Generates engineering-oriented recommendations.
    Always presented as 'Suggested engineering investigation', NOT confirmed root cause.
    """
    rec_data = ENGINEERING_RECOMMENDATION_DATABASE.get(predicted_class, ENGINEERING_RECOMMENDATION_DATABASE['none'])

    recommendation = {
        'suggested_investigation': rec_data['suggested_investigation'],
        'action_type': rec_data['action_type'],
        'target_investigation_areas': rec_data['target_areas'],
        'disclaimer': "Presented as 'Suggested engineering investigation', NOT confirmed root cause or guaranteed solution.",
        'recommendation_effectiveness_status': "Recommendation effectiveness cannot currently be measured from WM-811K.",
        'future_scope_recovery_model': {
            'description': "Recovery probability and intervention outcome measurement are identified as FUTURE SCOPE requiring manufacturing outcome logs.",
            'mathematical_formulation': "P(Recovery | Pattern, Process_State, Action) = f(W_defect, S_tool, A_intervention)"
        }
    }

    return recommendation
