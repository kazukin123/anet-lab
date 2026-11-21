package io.github.kazukin123.anetlab.metricsviewer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;

import io.github.kazukin123.anetlab.metricsviewer.service.WorkspaceManager;

/**
 * Metrics Viewer application entry point.
 */
@SpringBootApplication
public class MetricsViewerApplication {

	private static final Logger log = LoggerFactory.getLogger(MetricsViewerApplication.class);

	public static void main(String[] args) {
		log.info("Starting Metrics Viewer...");
		final ConfigurableApplicationContext context = SpringApplication.run(MetricsViewerApplication.class, args);
		final WorkspaceManager workspaceManager = context.getBean(WorkspaceManager.class);
		log.info("Workspaces directory: {}", workspaceManager.getWorkspacesDir());
		log.info("Current workspace: {} (runs: {})",
				workspaceManager.currentName(), workspaceManager.currentRunsDir());
		log.info("Metrics Viewer started successfully (port: 8080 by default)");
	}
}
