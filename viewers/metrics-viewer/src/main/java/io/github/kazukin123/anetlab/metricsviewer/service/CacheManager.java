package io.github.kazukin123.anetlab.metricsviewer.service;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import io.github.kazukin123.anetlab.metricsviewer.model.MetricsSnapshot;
import io.github.kazukin123.anetlab.metricsviewer.model.RunInfo;

/**
 * Handles Serializable cache load/save per run.
 */
@Component
public class CacheManager {

    private static final Logger log = LoggerFactory.getLogger(CacheManager.class);
    private static final String SNAPSHOT_FILE = "metrics_snapshot.ser";

    public Path getCachePath(RunInfo run) {
        return java.nio.file.Paths.get(run.getRunPath(), SNAPSHOT_FILE);
    }

    public MetricsSnapshot load(RunInfo run) {
        Path path = Paths.get(run.getRunPath(), SNAPSHOT_FILE);
        if (!Files.exists(path)) {
            log.debug("No cache file found for {}", run.getRunId());
            return null;
        }
        ObjectInputStream ois = null;
        try {
            ois = new ObjectInputStream(new BufferedInputStream(Files.newInputStream(path)));
            Object obj = ois.readObject();
            if (obj instanceof MetricsSnapshot) {
            	MetricsSnapshot snapshot = (MetricsSnapshot) obj;
                log.info("Loaded cache for run {}. {} tags. Position {}",
                		run.getRunId(), snapshot.getSeries().size(), snapshot.getLastReadPosition());
                return (MetricsSnapshot) obj;
            }
        } catch (Exception e) {
            log.warn("Cache load failed for {}: {}", run.getRunId(), e.getMessage());
            try {
                Files.deleteIfExists(path);
                log.warn("Deleted invalid cache file {}", path);
            } catch (IOException ex) {
                log.warn("Cache delete failed: {}", ex.getMessage());
            }
        } finally {
            if (ois != null) {
                try {
                    ois.close();
                } catch (IOException ignore) {
                }
            }
        }
        return null;
    }

    public void save(RunInfo run, MetricsSnapshot snapshot) {
        Path path = Paths.get(run.getRunPath(), SNAPSHOT_FILE);
        ObjectOutputStream oos = null;
        try {
            oos = new ObjectOutputStream(new BufferedOutputStream(Files.newOutputStream(path)));
            oos.writeObject(snapshot);
            log.debug("Saved cache for run {} ({} series)", run.getRunId(), snapshot.getSeries().size());
        } catch (IOException e) {
            log.warn("Cache save failed for {}: {}", run.getRunId(), e.getMessage());
        } finally {
            if (oos != null) {
                try {
                    oos.close();
                } catch (IOException ignore) {
                }
            }
        }
    }
}
