import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ConfigurationPanel } from './components/ConfigurationPanel';
import { AudioPlayer } from './components/AudioPlayer';
import { VisualizerTabs } from './components/Visualizers/VisualizerTabs';
import './index.css';

function App() {
  const { t, i18n } = useTranslation();
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  const [config, setConfig] = useState({
    composer: 'ludwig van beethoven',
    key: 'C_major',
    mood: 'custom',
    velocity: 50,
    density: 60,
    tempo: 120,
    duration: 30,
    temperature: 1.4,
    top_k: 25
  });

  const [isGenerating, setIsGenerating] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [midiUrl, setMidiUrl] = useState<string | null>(null);
  const [xmlUrl, setXmlUrl] = useState<string | null>(null);
  const [playbackTime, setPlaybackTime] = useState(0);

  const handleConfigChange = (key: keyof typeof config, value: string | number) => {
    setConfig(prev => {
      const next = { ...prev, [key]: value };

      // If mood changes, update sliders
      if (key === 'mood' && typeof value === 'string' && value !== 'custom') {
        if (value === 'cheerful') {
          next.velocity = 80; next.density = 70; next.tempo = 160;
        } else if (value === 'melancholic') {
          next.velocity = 40; next.density = 30; next.tempo = 60;
        } else if (value === 'energetic') {
          next.velocity = 90; next.density = 90; next.tempo = 180;
        } else if (value === 'peaceful') {
          next.velocity = 30; next.density = 40; next.tempo = 80;
        }
      } else if (['velocity', 'density', 'tempo'].includes(key)) {
        // If a slider changes natively, set mood to custom
        next.mood = 'custom';
      }
      return next;
    });
  };

  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      // Use configured API base URL instead of hardcoded localhost
      const response = await fetch(`${API_BASE_URL}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          composer: config.composer,
          key: config.key,
          velocity: config.velocity / 100, // normalize to 0-1 for backend
          density: config.density / 100,
          tempo: config.tempo / 100,
          duration: config.duration,
          temperature: config.temperature,
          top_k: config.top_k
        })
      });

      if (!response.ok) {
        throw new Error('Generation failed');
      }

      const data = await response.json();

      // Update URLs with dynamic API_BASE_URL prefix instead of hardcoded localhost
      if (data.audio_url) setAudioUrl(`${API_BASE_URL}${data.audio_url}`);
      if (data.midi_url) setMidiUrl(`${API_BASE_URL}${data.midi_url}`);
      if (data.xml_url) setXmlUrl(`${API_BASE_URL}${data.xml_url}`);

    } catch (error) {
      console.error(error);
      alert("Failed to generate music.");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <>
      {/* Left Sidebar */}
      <ConfigurationPanel
        config={config}
        onChange={handleConfigChange}
        onGenerate={handleGenerate}
        isGenerating={isGenerating}
      />

      {/* Main Content Area */}
      <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', minWidth: 0, position: 'relative' }}>
        <button
          onClick={() => i18n.changeLanguage(i18n.language.startsWith('zh') ? 'en' : 'zh')}
          style={{
            position: 'absolute', top: 0, right: 0,
            background: 'transparent', border: '1px solid var(--border-color)',
            color: 'var(--text-secondary)', padding: '4px 8px', borderRadius: '4px',
            cursor: 'pointer', fontSize: '12px', zIndex: 10
          }}
        >
          {i18n.language.startsWith('zh') ? 'EN' : '中文'}
        </button>
        <AudioPlayer title={t('player.title')} audioUrl={audioUrl} onTimeUpdate={setPlaybackTime} />
        <VisualizerTabs midiUrl={midiUrl} xmlUrl={xmlUrl} playbackTime={playbackTime} tempo={config.tempo} />
      </div>
    </>
  );
}

export default App;
