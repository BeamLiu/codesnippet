import React, { useEffect, useRef } from 'react';
import { OpenSheetMusicDisplay } from 'opensheetmusicdisplay';

interface Props {
    xmlUrl: string;
    playbackTime?: number;
    tempo?: number;
}

export const SheetMusicContainer: React.FC<Props> = ({ xmlUrl, playbackTime, tempo }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const osmdRef = useRef<OpenSheetMusicDisplay | null>(null);

    useEffect(() => {
        if (!containerRef.current) return;

        if (!osmdRef.current) {
            osmdRef.current = new OpenSheetMusicDisplay(containerRef.current, {
                autoResize: true,
                backend: "svg",
                drawTitle: false,
                drawSubtitle: false,
                drawComposer: false,
                drawLyricist: false,
                pageFormat: 'Endless'
            });

            // Apply dark mode colors
            osmdRef.current.EngravingRules.DefaultColorMusic = "#ffffff";
            osmdRef.current.EngravingRules.DefaultColorNotehead = "#ffffff";
            osmdRef.current.EngravingRules.DefaultColorStem = "#ffffff";
            osmdRef.current.EngravingRules.DefaultColorRest = "#ffffff";
            osmdRef.current.EngravingRules.DefaultColorLabel = "#ffffff";
            osmdRef.current.EngravingRules.DefaultColorTitle = "#ffffff";
            osmdRef.current.EngravingRules.StaffLineColor = "#a0a0ab";
        }

        const loadAndRender = async () => {
            try {
                await osmdRef.current?.load(xmlUrl);
                osmdRef.current?.render();
            } catch (err) {
                console.error("OSMD Load Error:", err);
            }
        };

        loadAndRender();

    }, [xmlUrl]);

    // Handle cursor tracking
    useEffect(() => {
        const osmd = osmdRef.current;
        if (!osmd || !osmd.cursor || playbackTime === undefined || !tempo) return;

        try {
            // Show cursor if not visible
            if (osmd.cursor.Hidden) {
                osmd.cursor.show();
            }

            // Number of seconds a whole note takes
            const secondsPerWholeNote = 240 / tempo;
            // The score's current elapsed "fraction" based on our physical playback time
            const targetFraction = playbackTime / secondsPerWholeNote;

            let it = osmd.cursor.iterator;
            if (!it) return;

            const currentFraction = it.currentTimeStamp.RealValue;

            // If user seeks backwards, completely reset cursor
            if (targetFraction < currentFraction) {
                osmd.cursor.reset();
                it = osmd.cursor.iterator;
            }

            // Walk cursor forward until it matches target time fraction
            while (it && it.currentTimeStamp.RealValue <= targetFraction && !it.EndReached) {
                osmd.cursor.next();
            }

            // Keep cursor in view automatically (page/line flipping)
            if (osmd.cursor.cursorElement && containerRef.current?.parentElement) {
                const scrollContainer = containerRef.current.parentElement;
                const cursorRect = osmd.cursor.cursorElement.getBoundingClientRect();
                const containerRect = scrollContainer.getBoundingClientRect();

                // If the cursor moves below or above the visible bounds
                if (cursorRect.bottom > containerRect.bottom || cursorRect.top < containerRect.top) {
                    scrollContainer.scrollBy({
                        top: cursorRect.top - containerRect.top - (containerRect.height / 3),
                        behavior: 'smooth'
                    });
                }
            }

        } catch (e) {
            console.error("Error advancing OSMD cursor:", e);
        }

    }, [playbackTime, tempo]);

    return (
        <div style={{ flexGrow: 1, backgroundColor: 'var(--bg-element)', width: '100%', overflowY: 'auto', padding: '16px', borderRadius: '8px' }}>
            <div ref={containerRef} style={{ width: '100%' }} />
        </div>
    );
};
