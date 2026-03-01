import React from 'react';
import 'html-midi-player';

interface Props {
    midiUrl: string;
}

export const PianoRollContainer: React.FC<Props> = ({ midiUrl }) => {
    const visualizerRef = React.useRef<HTMLElement>(null);

    React.useEffect(() => {
        if (visualizerRef.current) {
            // @ts-ignore
            visualizerRef.current.config = {
                noteRGB: '255, 255, 255',
                activeNoteRGB: '249, 115, 22' // accent-orange
            };
        }
    }, [midiUrl]);

    return (
        <div style={{ flexGrow: 1, backgroundColor: 'var(--bg-element)', overflow: 'auto', padding: '16px', borderRadius: '8px' }}>
            {/* @ts-ignore */}
            <midi-visualizer ref={visualizerRef} type="piano-roll" src={midiUrl}></midi-visualizer>
        </div>
    );
};
