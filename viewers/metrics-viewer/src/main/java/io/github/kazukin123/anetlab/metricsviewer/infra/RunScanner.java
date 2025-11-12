package io.github.kazukin123.anetlab.metricsviewer.infra;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class RunScanner {

	private static final Logger log = LoggerFactory.getLogger(RunScanner.class);
	private final Path runsDir;

	public RunScanner(@Value("${metricsviewer.runs-dir:runs}") String runsDirPath) {
		this.runsDir = Paths.get(runsDirPath);
	}

	public Path getRunsDir() {
		return runsDir;
	}

	public Path resolveRunDir(String runId) {
		return runsDir.resolve(runId);
	}

	private List<String> scanRunsDir() {
		final List<String> runs = new ArrayList<>();
		if (!Files.exists(runsDir)) {
			log.warn("Runs directory not found: {}", runsDir.toAbsolutePath());
			return runs; // 空リスト返却
		}

		try (Stream<Path> dirs = Files.list(runsDir)) {
			dirs.filter(Files::isDirectory)
					.sorted()
					.forEach(dir -> {
						final String runId = dir.getFileName().toString();
						runs.add(runId);
					});
		} catch (IOException e) {
			log.warn("Failed to scan runs directory: {}", e.getMessage());
		}
		return runs;
	}

	public List<String> listRunId() {
		return scanRunsDir();
	}
}
