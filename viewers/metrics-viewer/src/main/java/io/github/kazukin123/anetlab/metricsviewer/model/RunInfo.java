package io.github.kazukin123.anetlab.metricsviewer.model;

import java.nio.file.Path;

import lombok.Builder;
import lombok.Data;

/**
 * Run metadata and state (including differential read offset).
 */
@Data
@Builder
public class RunInfo {
    private String runId;
    private String name;
    private String runPath;
    private Path jsonlPath;
    private long lastUpdated;
    private int stepCount;
    private long lastReadPosition;
}
