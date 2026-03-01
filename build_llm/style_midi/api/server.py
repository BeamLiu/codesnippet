import sys
import os
import subprocess
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import StyleMIDIModel, ModelConfig
from tokenizer import REMITokenizer
from generate import generate_music

app = FastAPI(title="StyleMIDI API")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with exact frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer
tokenizer = None
model = None
device = "cpu"

class GenerateRequest(BaseModel):
    composer: str
    key: str
    velocity: float
    density: float
    tempo: float
    duration: float

@app.on_event("startup")
async def startup_event():
    global tokenizer, model, device
    print("Initializing tokenzier and model...")
    tokenizer = REMITokenizer()
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = StyleMIDIModel(config)
    
    # Try to load latest checkpoint if exists
    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../result'))
    latest_ckpt = None
    if os.path.exists(ckpt_dir):
        # find the most recent checkpoint. Example: model_step_7500.pt
        ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("model_step_") and f.endswith(".pt")]
        if ckpts:
            # Sort by step number
            ckpts.sort(key=lambda x: int(x.split('_')[2].split('.')[0]), reverse=True)
            latest_ckpt = os.path.join(ckpt_dir, ckpts[0])
            
    if latest_ckpt:
        print(f"Loading checkpoint {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: No checkpoint found in result/. Output will be noise.")
        
    model = model.to(device)
    
    # Ensure samples directory exists
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '../samples')), exist_ok=True)
    print("Initialization complete.")

@app.post("/api/generate")
async def generate_endpoint(req: GenerateRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not initialized")
        
    # Format conditions based on tokenizer vocab keys
    conds = {
        "COMPOSER": req.composer,
        "KEY": req.key,
        "VELOCITY": f"{req.velocity:.1f}",
        "DENSITY": f"{req.density:.1f}",
        "TEMPO": f"{req.tempo:.1f}"
    }
    
    try:
        print(f"Generating music with conditions: {conds}")
        midi_path = generate_music(
            model=model,
            tokenizer=tokenizer,
            conditions=conds,
            max_duration=req.duration,
            temperature=1.0,  # can expose this if needed
            top_p=0.9,
            device=device
        )
        
        # midi_path is relative inside generate_music (e.g. `../samples/generated_xxx.mid`)
        # Wait, generate_music returns the relative path `../samples/generated_...mid` which resolves relative to the CWD script is run.
        # But we also run server in `api/` or root?
        # generate.py hardcodes: out_path = f"../samples/generated_{conditions.get('COMPOSER', 'unknown')}_{conditions.get('MOOD', 'unknown')}.mid"
        # If server is run in root, it evaluates relative to cwd.
        
        abs_midi_path = os.path.abspath(midi_path)
        base_name = os.path.splitext(os.path.basename(abs_midi_path))[0]
        samples_dir = os.path.dirname(abs_midi_path)
        
        # For fluidsynth, it's better to render directly to mp3/wav
        # Use SoundFont if available: /usr/share/sounds/sf2/FluidR3_GM.sf2
        sf2_path = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
        mp3_filename = f"{base_name}.mp3"
        abs_mp3_path = os.path.join(samples_dir, mp3_filename)
        
        midi_filename = os.path.basename(abs_midi_path)
        
        # also generate musicxml using music21
        try:
            import music21
            score = music21.converter.parse(abs_midi_path)
            xml_filename = f"{base_name}.xml"
            abs_xml_path = os.path.join(samples_dir, xml_filename)
            score.write('musicxml', abs_xml_path)
        except Exception as e:
            print(f"Failed to generate musicxml: {e}")
            xml_filename = None
        
        if os.path.exists(sf2_path):
            # Fluidsynth rendering to direct wav then ffmpeg to mp3, or direct if supported.
            # fluidsynth -ni FluidR3_GM.sf2 input.mid -F output.wav -r 44100
            # Since fast rendering mp3 directly might not be supported without compile flags, we'll do wav->mp3 via ffmpeg if needed,
            # or just serve the .wav. Let's serve .wav for simplicity and speed, or mp3 if we have it.
            # actually we can just serve .wav
            wav_filename = f"{base_name}.wav"
            abs_wav_path = os.path.join(samples_dir, wav_filename)
            
            cmd = ["fluidsynth", "-ni", sf2_path, abs_midi_path, "-F", abs_wav_path, "-r", "44100"]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # optional: ffmpeg to mp3 to save space, but .wav is fine for local test
            # cmd_mp3 = ["ffmpeg", "-y", "-i", abs_wav_path, abs_mp3_path]
            # subprocess.run(cmd_mp3, check=True)
            
            return {
                "midi_url": f"/samples/{midi_filename}",
                "audio_url": f"/samples/{wav_filename}",
                "xml_url": f"/samples/{xml_filename}" if xml_filename else None
            }
        else:
            print("SoundFont not found, skipping audio rendering.")
            return {
                "midi_url": f"/samples/{midi_filename}",
                "audio_url": None,
                "xml_url": f"/samples/{xml_filename}" if xml_filename else None
            }
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Mount samples directory for static file serving
# This must be done after API routes or with a specific prefix
samples_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../samples'))
app.mount("/samples", StaticFiles(directory=samples_abspath), name="samples")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
