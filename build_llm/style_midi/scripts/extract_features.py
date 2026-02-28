import os
import json
import pandas as pd
import pretty_midi
import numpy as np
from tqdm import tqdm

def estimate_key(pm):
    total_pitch_class_histogram = np.zeros(12)
    for inst in pm.instruments:
        if not inst.is_drum:
            for note in inst.notes:
                total_pitch_class_histogram[note.pitch % 12] += (note.end - note.start)
                
    if np.sum(total_pitch_class_histogram) == 0:
        return "C_major"
        
    maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    keys = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    
    maj_corrs = [np.corrcoef(total_pitch_class_histogram, np.roll(maj_profile, i))[0, 1] for i in range(12)]
    min_corrs = [np.corrcoef(total_pitch_class_histogram, np.roll(min_profile, i))[0, 1] for i in range(12)]
    
    if np.max(maj_corrs) > np.max(min_corrs):
        return f"{keys[np.argmax(maj_corrs)]}_major"
    else:
        return f"{keys[np.argmax(min_corrs)]}_minor"

def calculate_audio_metrics(pm):
    # 1. Calculate Tempo
    try:
        tempo_bpm = pm.estimate_tempo()
    except:
        tempo_bpm = 120.0
        
    # Cap and normalize tempo (roughly 40 to 240 bpm -> 0.0 to 1.0)
    tempo_norm = max(0.0, min(1.0, (tempo_bpm - 40.0) / 200.0))
    # Bucketize into 10 steps (e.g. "0.0", "0.1", ... "1.0")
    tempo_val = f"{round(tempo_norm * 10) / 10:.1f}"

    # 2. Calculate Density and Velocity
    total_notes = 0
    total_velocity = 0
    duration = pm.get_end_time()
    for inst in pm.instruments:
        if not inst.is_drum:
            total_notes += len(inst.notes)
            total_velocity += sum(n.velocity for n in inst.notes)
            
    if duration == 0 or total_notes == 0:
        return {"tempo": tempo_val, "density": "0.0", "velocity": "0.0"}
        
    density_raw = total_notes / duration
    avg_vel_raw = total_velocity / total_notes
    
    # Cap and normalize density (0 to 15 notes/sec -> 0.0 to 1.0)
    density_norm = max(0.0, min(1.0, density_raw / 15.0))
    density_val = f"{round(density_norm * 10) / 10:.1f}"
    
    # Cap and normalize velocity (roughly 30 to 90 -> 0.0 to 1.0)
    vel_norm = max(0.0, min(1.0, (avg_vel_raw - 30.0) / 60.0))
    vel_val = f"{round(vel_norm * 10) / 10:.1f}"
    
    return {"tempo": tempo_val, "density": density_val, "velocity": vel_val}
