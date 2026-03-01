import React from 'react';
import * as Tabs from '@radix-ui/react-tabs';
import { useTranslation } from 'react-i18next';
import { PianoRollContainer } from './PianoRollContainer';
import { SheetMusicContainer } from './SheetMusicContainer';
import styles from './VisualizerTabs.module.css';

interface Props {
    midiUrl?: string | null;
    xmlUrl?: string | null;
    playbackTime?: number;
    tempo?: number;
}

export const VisualizerTabs: React.FC<Props> = ({ midiUrl, xmlUrl, playbackTime, tempo }) => {
    const { t } = useTranslation();

    return (
        <div className={styles.container}>
            <Tabs.Root className={styles.tabsRoot} defaultValue="sheet">
                <Tabs.List className={styles.tabsList} aria-label="Visualizers">
                    <Tabs.Trigger className={styles.tabsTrigger} value="sheet">
                        {t('visualizer.sheet_music')}
                    </Tabs.Trigger>
                    <Tabs.Trigger className={styles.tabsTrigger} value="piano">
                        {t('visualizer.piano_roll')}
                    </Tabs.Trigger>
                </Tabs.List>

                <div className={styles.contentContainer}>
                    <Tabs.Content className={styles.tabsContent} value="sheet">
                        {xmlUrl ? (
                            <SheetMusicContainer xmlUrl={xmlUrl} playbackTime={playbackTime} tempo={tempo} />
                        ) : (
                            <div className={styles.emptyState}>No music generated yet.</div>
                        )}
                    </Tabs.Content>
                    <Tabs.Content className={styles.tabsContent} value="piano">
                        {midiUrl ? (
                            <PianoRollContainer midiUrl={midiUrl} />
                        ) : (
                            <div className={styles.emptyState}>No music generated yet.</div>
                        )}
                    </Tabs.Content>
                </div>
            </Tabs.Root>
        </div>
    );
};
