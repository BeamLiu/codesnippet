import React from 'react';
import * as Select from '@radix-ui/react-select';
import * as RadioGroup from '@radix-ui/react-radio-group';
import * as Slider from '@radix-ui/react-slider';
import * as Dialog from '@radix-ui/react-dialog';
import { ChevronDown, Settings } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import styles from './ConfigurationPanel.module.css';

interface ConfigState {
    composer: string;
    key: string;
    mood: string;
    velocity: number;
    density: number;
    tempo: number;
    duration: number;
    temperature: number;
    top_k: number;
}

interface Props {
    config: ConfigState;
    onChange: (key: keyof ConfigState, value: string | number) => void;
    onGenerate: () => void;
    isGenerating: boolean;
}

const COMPOSERS = [
    { value: 'beethoven', label: 'Beethoven' },
    { value: 'chopin', label: 'Chopin' },
    { value: 'mozart', label: 'Mozart' },
    { value: 'schubert', label: 'Schubert' }
];

const KEYS = [
    { value: 'C_major', label: 'C Major' },
    { value: 'A_minor', label: 'A Minor' },
    { value: 'G_major', label: 'G Major' },
    { value: 'F_major', label: 'F Major' },
];

export const ConfigurationPanel: React.FC<Props> = ({ config, onChange, onGenerate, isGenerating }) => {
    const { t } = useTranslation();

    const [composers, setComposers] = React.useState(COMPOSERS);
    const [keys, setKeys] = React.useState(KEYS);

    React.useEffect(() => {
        fetch('/enums.json')
            .then(res => res.json())
            .then(data => {
                if (data.COMPOSER) {
                    setComposers(data.COMPOSER.map((c: string) => ({ value: c, label: c.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) })));
                }
                if (data.KEY) {
                    setKeys(data.KEY.map((k: string) => ({ value: k, label: k.replace(/_/g, ' ') })));
                }
            })
            .catch(err => console.error("Could not load enums.json", err));
    }, []);

    const handleSliderChange = (key: keyof ConfigState) => (val: number[]) => {
        onChange(key, val[0]);
    };

    return (
        <div className={styles.panel}>
            <div className={styles.header}>
                <div className={styles.logo}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <rect x="2" y="8" width="3" height="8" rx="1.5" fill="var(--accent-orange)" />
                        <rect x="7" y="4" width="3" height="16" rx="1.5" fill="var(--accent-orange)" />
                        <rect x="12" y="2" width="3" height="20" rx="1.5" fill="var(--accent-orange)" />
                        <rect x="17" y="6" width="3" height="12" rx="1.5" fill="var(--accent-orange)" />
                        <rect x="22" y="10" width="3" height="4" rx="1.5" fill="var(--accent-orange)" />
                    </svg>
                    <h1>StyleMIDI</h1>
                </div>
                <Dialog.Root>
                    <Dialog.Trigger asChild>
                        <button className={styles.advancedBtn} aria-label="Advanced Settings">
                            <Settings size={18} />
                        </button>
                    </Dialog.Trigger>
                    <Dialog.Portal>
                        <Dialog.Overlay className={styles.dialogOverlay} />
                        <Dialog.Content className={styles.dialogContent}>
                            <Dialog.Title className={styles.dialogTitle}>{t('config.advanced')}</Dialog.Title>
                            <Dialog.Description className={styles.dialogDesc}>
                                {t('config.advanced_desc')}
                            </Dialog.Description>

                            <CustomSlider
                                label={t('config.temperature')}
                                value={config.temperature}
                                onChange={handleSliderChange('temperature')}
                                max={2}
                                step={0.1}
                            />
                            <CustomSlider
                                label={t('config.top_k')}
                                value={config.top_k}
                                onChange={handleSliderChange('top_k')}
                                max={100}
                                step={1}
                            />
                        </Dialog.Content>
                    </Dialog.Portal>
                </Dialog.Root>
            </div>

            <div className={styles.sectionTitle}>{t('config.title').toUpperCase()}</div>

            {/* Selectors */}
            <div className={styles.selectors}>
                <CustomSelect
                    value={config.composer}
                    onChange={(v: string) => onChange('composer', v)}
                    options={composers}
                    placeholder={t('config.composer')}
                />
                <CustomSelect
                    value={config.key}
                    onChange={(v: string) => onChange('key', v)}
                    options={keys}
                    placeholder={t('config.key')}
                />
            </div>

            {/* Mood Radios */}
            <div className={styles.inputGroup}>
                <label className={styles.inputLabel}>{t('config.mood')}</label>
                <RadioGroup.Root
                    className={styles.radioGroup}
                    value={config.mood}
                    onValueChange={(v) => onChange('mood', v)}
                >
                    {['cheerful', 'melancholic', 'energetic'].map(m => (
                        <div key={m} className={styles.radioItem}>
                            <RadioGroup.Item className={styles.radioIndicator} value={m} id={m}>
                                <RadioGroup.Indicator className={styles.radioIndicatorInner} />
                            </RadioGroup.Item>
                            <label className={styles.radioLabel} htmlFor={m}>{t(`config.mood_${m}`)}</label>
                        </div>
                    ))}
                </RadioGroup.Root>
            </div>

            {/* Sliders */}
            <CustomSlider label={t('config.velocity')} value={config.velocity} onChange={handleSliderChange('velocity')} max={100} unit="%" bottomLabel="Med" />
            <CustomSlider label={t('config.density')} value={config.density} onChange={handleSliderChange('density')} max={110} displayValue={`${config.density} / 110`} bottomLabel="5 units" />
            <CustomSlider label={t('config.tempo')} value={config.tempo} onChange={handleSliderChange('tempo')} max={240} unit=" bpm" bottomLabel="Allegro" />
            <CustomSlider label={t('config.duration')} value={config.duration} onChange={handleSliderChange('duration')} max={120} displayValue={`${config.duration} s`} bottomLabel="Full Track" />

            <button className={styles.generateButton} onClick={onGenerate} disabled={isGenerating}>
                {isGenerating ? t('config.generating') : t('config.generate')}
            </button>
        </div>
    );
};

// --- Custom Subcomponents using Radix Primitives ---

const CustomSelect = ({ value, onChange, options, placeholder }: any) => (
    <Select.Root value={value} onValueChange={onChange}>
        <Select.Trigger className={styles.selectTrigger} aria-label={placeholder}>
            <Select.Value placeholder={placeholder} />
            <Select.Icon className={styles.selectIcon}>
                <ChevronDown size={16} />
            </Select.Icon>
        </Select.Trigger>

        <Select.Portal>
            <Select.Content className={styles.selectContent}>
                <Select.ScrollUpButton className={styles.selectScrollButton} />
                <Select.Viewport className={styles.selectViewport}>
                    {options.map((opt: any) => (
                        <Select.Item key={opt.value} value={opt.value} className={styles.selectItem}>
                            <Select.ItemText>{opt.label}</Select.ItemText>
                        </Select.Item>
                    ))}
                </Select.Viewport>
                <Select.ScrollDownButton className={styles.selectScrollButton} />
            </Select.Content>
        </Select.Portal>
    </Select.Root>
);

const CustomSlider = ({ label, value, onChange, max, step = 1, unit = "", displayValue, bottomLabel }: any) => (
    <div className={styles.sliderContainer}>
        <div className={styles.sliderHeader}>
            <span className={styles.sliderLabel}>{label}</span>
            <span className={styles.sliderValue}>{displayValue || `${value}${unit}`}</span>
        </div>
        <Slider.Root className={styles.sliderRoot} value={[value]} max={max} step={step} onValueChange={onChange}>
            <Slider.Track className={styles.sliderTrack}>
                <Slider.Range className={styles.sliderRange} />
            </Slider.Track>
            <Slider.Thumb className={styles.sliderThumb} aria-label={label} />
        </Slider.Root>
        {bottomLabel && <div className={styles.sliderBottomLabel}>{bottomLabel}</div>}
    </div>
);
