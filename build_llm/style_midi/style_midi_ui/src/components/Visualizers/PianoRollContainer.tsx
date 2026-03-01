import React, { useEffect, useState, useRef } from 'react';
import * as mm from '@magenta/music/es6/core';
import styles from './PianoRoll.module.css';

interface Props {
    midiUrl: string;
    playbackTime?: number;
}

// Colors derived precisely from the requested design mockup pattern
const COLORS = [
    '#3b82f6', // Blue
    '#a855f7', // Purple
    '#f97316', // Orange
    '#22c55e', // Green
    '#ec4899', // Pink
    '#eab308'  // Yellow
];

const PITCH_HEIGHT = 16;
const PIXELS_PER_SECOND = 100; // 100px translates to 1 second. A generous scroll width

export const PianoRollContainer: React.FC<Props> = ({ midiUrl, playbackTime = 0 }) => {
    const [notes, setNotes] = useState<any[]>([]);
    const [minPitch, setMinPitch] = useState(21);
    const [maxPitch, setMaxPitch] = useState(108);
    const [totalDuration, setTotalDuration] = useState(0);

    const trackRef = useRef<HTMLDivElement>(null);
    const keyboardRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        let active = true;
        mm.urlToNoteSequence(midiUrl).then(ns => {
            if (!active) return;
            const nsNotes = ns.notes || [];
            setNotes(nsNotes);

            if (nsNotes.length > 0) {
                let minP = 127;
                let maxP = 0;
                let maxD = 0;
                nsNotes.forEach(n => {
                    const p = n.pitch || 0;
                    const et = n.endTime || 0;
                    if (p < minP) minP = p;
                    if (p > maxP) maxP = p;
                    if (et > maxD) maxD = et;
                });

                // Add margins top and bottom ensuring we round out to the nearest octave borders for aesthetics
                minP = Math.max(21, minP - 5);
                maxP = Math.min(108, maxP + 5);
                setMinPitch(minP);
                setMaxPitch(maxP);
                setTotalDuration(maxD);

                // Auto center scroll on init
                setTimeout(() => {
                    if (trackRef.current && keyboardRef.current) {
                        const initScroll = ((maxP - maxPitch) + (maxPitch - minP) / 2) * PITCH_HEIGHT;
                        trackRef.current.scrollTop = initScroll;
                        keyboardRef.current.scrollTop = initScroll;
                    }
                }, 100);
            }
        }).catch(err => console.error(err));

        return () => { active = false; };
    }, [midiUrl]);

    // Track cursor auto-scroll (horizontal bounds tracking)
    useEffect(() => {
        if (trackRef.current) {
            const currentX = playbackTime * PIXELS_PER_SECOND;
            const viewWidth = trackRef.current.clientWidth;
            const scrollLeft = trackRef.current.scrollLeft;

            if (currentX > scrollLeft + viewWidth * 0.8 || currentX < scrollLeft) {
                trackRef.current.scrollTo({
                    left: Math.max(0, currentX - viewWidth * 0.2),
                    behavior: 'smooth'
                });
            }
        }
    }, [playbackTime]);

    // Safely sync vertical scrolling of track across to the non-scrolling Piano keyboard panel
    const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
        if (keyboardRef.current) {
            keyboardRef.current.scrollTop = e.currentTarget.scrollTop;
        }
    };

    const pitches = [];
    for (let p = maxPitch; p >= minPitch; p--) {
        pitches.push(p);
    }

    // MIDI intervals modulo 12 definition
    const isBlack = (pitch: number) => {
        const p = pitch % 12;
        return [1, 3, 6, 8, 10].includes(p);
    };

    return (
        <div className={styles.container}>
            {/* Locked vertical Piano Keyboard Sidebar */}
            <div className={styles.keyboard} ref={keyboardRef}>
                {pitches.map(p => {
                    const black = isBlack(p);
                    return (
                        <div
                            key={p}
                            className={`${styles.keyRow} ${black ? styles.keyBlack : styles.keyWhite}`}
                        >
                        </div>
                    );
                })}
            </div>

            {/* X and Y Scrollable Notation Canvas Area */}
            <div className={styles.trackScroll} ref={trackRef} onScroll={handleScroll}>
                <div
                    className={styles.trackContent}
                    style={{
                        width: Math.max(800, totalDuration * PIXELS_PER_SECOND + 100),
                        height: pitches.length * PITCH_HEIGHT
                    }}
                >
                    {/* Underlying Grid Rendering */}
                    <div className={styles.gridBg}>
                        {pitches.map(p => (
                            <div key={`grid-${p}`} className={`${styles.gridLine} ${isBlack(p) ? styles.gridLineDark : ''}`} />
                        ))}
                    </div>

                    {/* Extracted Note Sequence rendering with colorful scheme matching layout constraints */}
                    {notes.map((n, i) => {
                        const p = n.pitch || 0;
                        const top = (maxPitch - p) * PITCH_HEIGHT;
                        const left = (n.startTime || 0) * PIXELS_PER_SECOND;
                        const width = ((n.endTime || 0) - (n.startTime || 0)) * PIXELS_PER_SECOND;
                        const color = COLORS[p % COLORS.length]; // cyclical vibrant rainbow modulo mapping

                        return (
                            <div
                                key={i}
                                className={styles.note}
                                style={{
                                    top,
                                    left,
                                    width: Math.max(4, width), // Provide minimum rendering size buffer
                                    backgroundColor: color
                                }}
                            />
                        );
                    })}

                    {/* Progress Cursor Handle */}
                    <div className={styles.playhead} style={{ left: playbackTime * PIXELS_PER_SECOND }} />
                </div>
            </div>
        </div>
    );
};
