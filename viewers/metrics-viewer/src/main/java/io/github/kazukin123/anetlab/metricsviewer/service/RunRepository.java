package io.github.kazukin123.anetlab.metricsviewer.service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import io.github.kazukin123.anetlab.metricsviewer.model.Run;

@Component
public class RunRepository {

    private static final Logger log = LoggerFactory.getLogger(RunRepository.class);
    private final Path runsDir;

    public RunRepository(@Value("${metricsviewer.runs-dir:runs}") String runsDirPath) {
        this.runsDir = Paths.get(runsDirPath);
    }

    public Path getRunsDir() {
        return runsDir;
    }

    public Path resolveRunDir(String runId) {
        return runsDir.resolve(runId);
    }

    private List<Run> scanRuns() {
        List<Run> runs = new ArrayList<>();
        if (!Files.exists(runsDir)) {
            log.warn("Runs directory not found: {}", runsDir.toAbsolutePath());
            return runs; // 空リスト返却
        }

        try (Stream<Path> dirs = Files.list(runsDir)) {
            dirs.filter(Files::isDirectory)
                .sorted()
                .forEach(dir -> {
                    String runId = dir.getFileName().toString();
                    runs.add(Run.builder().id(runId).tags(Collections.emptyList()).build());
                });
        } catch (IOException e) {
            log.warn("Failed to scan runs directory: {}", e.getMessage());
        }
        return runs;
    }

    public List<Run> getRuns() {
        // 空でもnullは返さない
        return scanRuns();
    }
}
