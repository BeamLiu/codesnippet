import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from tokenizer import REMITokenizer

def test_tokenizer_smoke():
    t = REMITokenizer()
    print(f"Tokenizer ready. Vocab size: {t.vocab_size}")
    
    sample_midi = os.path.abspath(os.path.join(os.path.dirname(__file__), '../samples/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'))
    
    if os.path.exists(sample_midi):
        print(f"Found sample MIDI: {sample_midi}")
        conds = {"COMPOSER": "beethoven", "MOOD": "energetic", "TEMPO": "allegro", "KEY": "C_minor"}
        print("Encoding...")
        tokens = t.encode(sample_midi, conditions=conds)
        print(f"Encoded into {len(tokens)} tokens.")
        
        print("Decoding...")
        decoded_pm = t.decode(tokens)
        
        out_path = os.path.join(os.path.dirname(__file__), '../samples/test_tokenizer_decoded.mid')
        decoded_pm.write(out_path)
        print(f"Decoded MIDI saved to: {out_path}")
        
        assert len(tokens) > 10, "Should have encoded some note events."
    else:
        print(f"Sample MIDI not found at {sample_midi}. Skipping real encode test.")

if __name__ == "__main__":
    test_tokenizer_smoke()
