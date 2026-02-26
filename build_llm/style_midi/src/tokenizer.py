import os
from typing import List, Dict, Optional, Tuple
import pretty_midi

class REMITokenizer:
    """
    REMI (Revamped MIDI) Tokenizer for StyleMIDI.
    Converts MIDI files to discrete token sequences and vice versa.
    
    Tokens:
    - Special: [PAD], [BOS], [EOS]
    - Note events: BAR, POSITION, PITCH, DURATION, VELOCITY
    - Conditions: COMPOSER, MOOD, TEMPO, KEY
    """
    def __init__(self, ticks_per_beat: int = 480):
        self.ticks_per_beat = ticks_per_beat
        self.positions_per_bar = 48    # Quantize a 4/4 bar into 48 steps
        
        # Duration mapping: fractions of a quarter note
        # 16 duration bins
        self.duration_bins = [
            0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 
            2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0
        ]
        
        # Build Vocab
        self.vocab: Dict[str, int] = {}
        self.idx_to_str: Dict[int, str] = {}
        
        self._build_vocab()
        print(f"Vocab size: {len(self.vocab)}")
        
    def _add_token(self, token_str: str):
        if token_str not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token_str] = idx
            self.idx_to_str[idx] = token_str

    def _build_vocab(self):
        # 1. Special tokens
        special_tokens = ["PAD", "BOS", "EOS"]
        for t in special_tokens:
            self._add_token(f"[{t}]")
            
        # 2. Condition tokens (Prefix)
        composers = ["beethoven", "chopin", "mozart", "schubert"]
        moods = ["energetic", "lyrical", "melancholic", "playful"]
        tempos = ["largo", "andante", "moderato", "allegro", "presto"]
        keys = ["C_major", "G_major", "D_major", "A_major", "E_major", "B_major", "F#_major", "Db_major", "Ab_major", "Eb_major", "Bb_major", "F_major",
                "A_minor", "E_minor", "B_minor", "F#_minor", "C#_minor", "G#_minor", "D#_minor", "Bb_minor", "F_minor", "C_minor", "G_minor", "D_minor"]
        
        for pfix, items in [("COMPOSER", composers), ("MOOD", moods), ("TEMPO", tempos), ("KEY", keys)]:
            for item in items:
                self._add_token(f"[{pfix}:{item}]")

        # 3. Note Event Tokens
        self._add_token("Bar_None")
        
        for i in range(self.positions_per_bar):
            self._add_token(f"Position_{i}")
            
        for i in range(128):  # MIDI PITCH
            self._add_token(f"Pitch_{i}")
            
        for i in range(32):   # VELOCITY (quantized to 32 bins, 0-31 * 4)
            self._add_token(f"Velocity_{i}")
            
        for i in range(len(self.duration_bins)):
            self._add_token(f"Duration_{i}")

    @property
    def pad_token_id(self) -> int:
        return self.vocab["[PAD]"]
        
    @property
    def bos_token_id(self) -> int:
        return self.vocab["[BOS]"]
        
    @property
    def eos_token_id(self) -> int:
        return self.vocab["[EOS]"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _quantize_velocity(self, vel: int) -> int:
        return max(0, min(31, vel // 4))

    def _find_closest_duration(self, duration_beats: float) -> int:
        diffs = [abs(duration_beats - b) for b in self.duration_bins]
        return diffs.index(min(diffs))

    def encode(self, midi_path: str, conditions: Optional[Dict[str, str]] = None) -> List[int]:
        """
        Encode a MIDI file to a list of token IDs.
        Assumes 4/4 time signature for simplicity.
        """
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            print(f"Failed to load {midi_path}: {e}")
            return []
            
        # Get tempo to convert time to beats
        # Simplify: assume constant tempo, use the first tempo change
        tempos = pm.get_tempo_changes()
        tempo = tempos[1][0] if len(tempos[1]) > 0 else 120.0
        sec_per_beat = 60.0 / tempo
        beats_per_bar = 4.0 # default 4/4
        
        # Collect all notes from all instruments
        all_notes = []
        for inst in pm.instruments:
            if not inst.is_drum:
                for note in inst.notes:
                    # convert time to beats
                    start_beat = note.start / sec_per_beat
                    end_beat = note.end / sec_per_beat
                    dur_beat = end_beat - start_beat
                    
                    bar = int(start_beat // beats_per_bar)
                    pos_in_bar_beats = start_beat % beats_per_bar
                    pos_idx = int((pos_in_bar_beats / beats_per_bar) * self.positions_per_bar)
                    pos_idx = max(0, min(self.positions_per_bar - 1, pos_idx))
                    
                    all_notes.append({
                        'bar': bar,
                        'pos': pos_idx,
                        'pitch': note.pitch,
                        'vel': self._quantize_velocity(note.velocity),
                        'dur_idx': self._find_closest_duration(dur_beat)
                    })
                    
        # Sort notes by bar, then position, then pitch
        all_notes.sort(key=lambda x: (x['bar'], x['pos'], x['pitch']))
        
        # Build token list
        tokens = []
        
        # Add Conditions Prefix
        tokens.append(self.bos_token_id)
        if conditions:
            for k, v in conditions.items():
                tok_str = f"[{k}:{v}]"
                if tok_str in self.vocab:
                    tokens.append(self.vocab[tok_str])
                    
        # Add Note Events
        current_bar = -1
        current_pos = -1
        
        for note in all_notes:
            if note['bar'] > current_bar:
                tokens.append(self.vocab["Bar_None"])
                current_bar = note['bar']
                current_pos = -1 # reset pos in new bar
                
            if note['pos'] != current_pos:
                tokens.append(self.vocab[f"Position_{note['pos']}"])
                current_pos = note['pos']
                
            tokens.append(self.vocab[f"Pitch_{note['pitch']}"])
            tokens.append(self.vocab[f"Velocity_{note['vel']}"])
            tokens.append(self.vocab[f"Duration_{note['dur_idx']}"])
            
        tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: List[int], initial_tempo: float = 120.0) -> pretty_midi.PrettyMIDI:
        """
        Decode a list of token IDs back into a PrettyMIDI object.
        """
        pm = pretty_midi.PrettyMIDI(initial_tempo=initial_tempo)
        inst = pretty_midi.Instrument(program=0) # Acoustic Grand Piano
        
        sec_per_beat = 60.0 / initial_tempo
        beats_per_bar = 4.0
        sec_per_bar = beats_per_bar * sec_per_beat
        
        current_bar = 0
        current_time_sec = 0.0
        
        i = 0
        while i < len(token_ids):
            tid = token_ids[i]
            t_str = self.idx_to_str.get(tid, "")
            
            if t_str == "Bar_None":
                current_bar += 1
                i += 1
                continue
                
            if t_str.startswith("Position_"):
                pos = int(t_str.split("_")[1])
                pos_ratio = pos / self.positions_per_bar
                current_time_sec = (current_bar * sec_per_bar) + (pos_ratio * sec_per_bar)
                i += 1
                continue
                
            if t_str.startswith("Pitch_"):
                # We expect Pitch, Vel, Dur in sequence
                if i + 2 < len(token_ids):
                    vel_str = self.idx_to_str.get(token_ids[i+1], "")
                    dur_str = self.idx_to_str.get(token_ids[i+2], "")
                    
                    if vel_str.startswith("Velocity_") and dur_str.startswith("Duration_"):
                        pitch = int(t_str.split("_")[1])
                        vel_idx = int(vel_str.split("_")[1])
                        dur_idx = int(dur_str.split("_")[1])
                        
                        velocity = min(127, vel_idx * 4 + 2) # restore approx vel
                        dur_beats = self.duration_bins[dur_idx]
                        dur_sec = dur_beats * sec_per_beat
                        
                        note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=current_time_sec,
                            end=current_time_sec + dur_sec
                        )
                        inst.notes.append(note)
                        
                        i += 3 # processed 3 tokens
                        continue
                        
            # Skip unknown/condition tokens
            i += 1
            
        pm.instruments.append(inst)
        return pm


