import React, { useRef, useState, useEffect } from 'react';
import { Play, Pause } from 'lucide-react';
import * as Slider from '@radix-ui/react-slider';
import styles from './AudioPlayer.module.css';

interface Props {
    title: string;
    audioUrl?: string | null;
    onTimeUpdate?: (time: number) => void;
}

export const AudioPlayer: React.FC<Props> = ({ title, audioUrl, onTimeUpdate }) => {
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);

    useEffect(() => {
        if (audioUrl && audioRef.current) {
            audioRef.current.load();
            setIsPlaying(false);
            setCurrentTime(0);
        }
    }, [audioUrl]);

    const togglePlay = () => {
        if (!audioRef.current || !audioUrl) return;
        if (isPlaying) {
            audioRef.current.pause();
        } else {
            audioRef.current.play();
        }
        setIsPlaying(!isPlaying);
    };

    const handleTimeUpdate = () => {
        if (audioRef.current) {
            setCurrentTime(audioRef.current.currentTime);
            onTimeUpdate?.(audioRef.current.currentTime);
        }
    };

    const handleLoadedMetadata = () => {
        if (audioRef.current) {
            setDuration(audioRef.current.duration);
        }
    };

    const handleSliderChange = (val: number[]) => {
        if (audioRef.current) {
            audioRef.current.currentTime = val[0];
            setCurrentTime(val[0]);
            onTimeUpdate?.(val[0]);
        }
    };

    const handleEnded = () => {
        setIsPlaying(false);
        setCurrentTime(0);
    };

    const formatTime = (time: number) => {
        if (isNaN(time)) return "0:00";
        const mins = Math.floor(time / 60);
        const secs = Math.floor(time % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <div className={styles.container}>
            <h2 className={styles.title}>{title}</h2>

            <div className={styles.playerControls}>
                <button className={styles.playButton} onClick={togglePlay} disabled={!audioUrl}>
                    {isPlaying ? <Pause size={20} fill="currentColor" /> : <Play size={20} fill="currentColor" className={styles.playIcon} />}
                </button>

                <div className={styles.scrubberContainer}>
                    <Slider.Root
                        className={styles.sliderRoot}
                        value={[currentTime]}
                        max={duration || 100}
                        step={0.1}
                        onValueChange={handleSliderChange}
                        disabled={!audioUrl}
                    >
                        <Slider.Track className={styles.sliderTrack}>
                            <Slider.Range className={styles.sliderRange} />
                        </Slider.Track>
                        <Slider.Thumb className={styles.sliderThumb} aria-label="Audio progress" />
                    </Slider.Root>

                    <div className={styles.timeInfo}>
                        <span>{formatTime(currentTime)}</span>
                        <span>{formatTime(duration)}</span>
                    </div>
                </div>
            </div>

            {audioUrl && (
                <audio
                    ref={audioRef}
                    src={audioUrl}
                    onTimeUpdate={handleTimeUpdate}
                    onLoadedMetadata={handleLoadedMetadata}
                    onEnded={handleEnded}
                />
            )}
        </div>
    );
};
