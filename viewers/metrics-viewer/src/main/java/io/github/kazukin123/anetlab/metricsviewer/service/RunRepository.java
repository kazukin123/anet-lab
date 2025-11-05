package io.github.kazukin123.anetlab.metricsviewer.service;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import io.github.kazukin123.anetlab.metricsviewer.model.RunInfo;

/**
 * Repository scanning 'runs/' directory and managing RunInfo list.
 */
@Component
public class RunRepository {

    private static final Logger log = LoggerFactory.getLogger(RunRepository.class);

    /** Root directory for runs. Default is "./runs". */
    private final Path runsRoot = Paths.get("runs");

    private final Map<String, RunInfo> runs = new LinkedHashMap<>();

    public void scanRuns() {
        runs.clear();
        if (!Files.exists(runsRoot) || !Files.isDirectory(runsRoot)) {
            log.warn("Runs directory does not exist: {}", runsRoot.toAbsolutePath());
            return;
        }
        int count = 0;
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(runsRoot)) {
            for (Path p : ds) {
                if (!Files.isDirectory(p)) continue;
                Path jsonl = p.resolve("metrics.jsonl");
                if (!Files.exists(jsonl)) continue;

                String runId = p.getFileName().toString();
                long lastUpdated = Files.getLastModifiedTime(jsonl).toMillis();
                RunInfo info = RunInfo.builder()
                        .runId(runId)
                        .name(runId)
                        .runPath(p.toAbsolutePath().toString())
                        .jsonlPath(jsonl.toAbsolutePath())
                        .lastUpdated(lastUpdated)
                        .stepCount(0)
                        .lastReadPosition(0L)
                        .build();
                runs.put(runId, info);
                count++;
            }
        } catch (IOException e) {
            log.error("Failed to scan runs directory: {}", e.getMessage(), e);
        }
        log.info("Run scan complete: {} run(s) detected under {}", count, runsRoot.toAbsolutePath());
    }

    public List<RunInfo> getAll() {
        return new ArrayList<RunInfo>(runs.values());
    }

    public Optional<RunInfo> find(String runId) {
        return Optional.ofNullable(runs.get(runId));
    }
}
