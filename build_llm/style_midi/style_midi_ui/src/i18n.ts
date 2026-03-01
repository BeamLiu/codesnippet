import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

const resources = {
    en: {
        translation: {
            "app": {
                "title": "StyleMIDI Engine",
                "description": "Create stylized MIDI music from scratch using Transformer. Set your conditions and hit generate.",
            },
            "config": {
                "title": "Configuration",
                "composer": "Composer",
                "key": "Key",
                "mood": "Mood",
                "mood_cheerful": "Cheerful",
                "mood_melancholic": "Melancholic",
                "mood_energetic": "Energetic",
                "mood_peaceful": "Peaceful",
                "mood_custom": "Custom",
                "velocity": "Velocity",
                "density": "Density",
                "tempo": "Tempo",
                "duration": "Duration",
                "generate": "Generate Music",
                "generating": "Generating..."
            },
            "player": {
                "title": "Audio Player",
            },
            "visualizer": {
                "sheet_music": "Sheet Music",
                "piano_roll": "Piano Roll"
            }
        }
    },
    zh: {
        translation: {
            "app": {
                "title": "StyleMIDI 引擎",
                "description": "从零实现 Transformer 的风格化作曲，设置您的条件并点击生成。",
            },
            "config": {
                "title": "生成控制面板",
                "composer": "作曲家",
                "key": "调性",
                "mood": "情绪预设",
                "mood_cheerful": "欢快",
                "mood_melancholic": "忧郁",
                "mood_energetic": "激昂",
                "mood_peaceful": "宁静",
                "mood_custom": "自定义",
                "velocity": "力度",
                "density": "音符密度",
                "tempo": "速度",
                "duration": "生成时长",
                "generate": "一键生成音乐",
                "generating": "生成中..."
            },
            "player": {
                "title": "试听",
            },
            "visualizer": {
                "sheet_music": "五线谱演示",
                "piano_roll": "钢琴卷帘"
            }
        }
    }
};

i18n
    .use(LanguageDetector)
    .use(initReactI18next)
    .init({
        resources,
        fallbackLng: "en",
        interpolation: {
            escapeValue: false
        }
    });

export default i18n;
