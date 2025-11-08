package io.github.kazukin123.anetlab.metricsviewer.service;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import io.github.kazukin123.anetlab.metricsviewer.model.Run;

/**
 * Repository scanning 'runs/' directory and managing RunInfo list.
 */
@Component
public class RunRepository {

    private static final Logger log = LoggerFactory.getLogger(RunRepository.class);

    private final Path runsDir;
    
    public RunRepository(@Value("${metricsviewer.runs-dir:runs}") String runsDirPath) {
        this.runsDir = Paths.get(runsDirPath);
    }

    private List<Run> scanRuns() {
		// TODO 自動生成されたメソッド・スタブ
    	return null;
    }

    public List<Run> getRuns() {
		// TODO 自動生成されたメソッド・スタブ
    	return null;
    }

}
