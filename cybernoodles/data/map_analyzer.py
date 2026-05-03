import os
import glob
import json
from cybernoodles.data.dataset_builder import get_map_notes

MAPS_DIR = os.path.join("data", "maps")

def analyze_all_maps():
    maps = glob.glob(os.path.join(MAPS_DIR, "*.zip"))
    curriculum = []
    
    print(f"Analyzing {len(maps)} maps for curriculum...")
    
    for m in maps:
        map_hash = os.path.basename(m).replace('.zip', '')
        notes, bpm = get_map_notes(map_hash)
        
        if not notes or len(notes) < 2:
            continue
        scorable_notes = [note for note in notes if int(note.get('type', -1)) != 3]
        if len(scorable_notes) < 2:
            continue
             
        # Calculate duration in seconds
        # time is in beats. duration = (last_beat - first_beat) / (bpm / 60)
        duration = (scorable_notes[-1]['time'] - scorable_notes[0]['time']) / (bpm / 60.0)
        if duration <= 0: continue
        
        nps = len(scorable_notes) / duration
        
        curriculum.append({
            'hash': map_hash,
            'nps': nps,
            'note_count': len(scorable_notes)
        })
        
    # Sort by NPS (easiest first)
    curriculum = sorted(curriculum, key=lambda x: x['nps'])
    
    with open('curriculum.json', 'w') as f:
        json.dump(curriculum, f, indent=4)
        
    print(f"Academy Graduated! curriculum.json created with {len(curriculum)} ranked maps.")
    if curriculum:
        print(f"Easiest: {curriculum[0]['nps']:.2f} NPS")
        print(f"Hardest: {curriculum[-1]['nps']:.2f} NPS")

if __name__ == "__main__":
    analyze_all_maps()
