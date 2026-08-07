package io.github.kazukin123.anetlab.metricsviewer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Metrics Viewer application entry point.
 */
@SpringBootApplication
public class MetricsViewerApplication {

	private static final Logger log = LoggerFactory.getLogger(MetricsViewerApplication.class);

	public static void main(String[] args) {
		log.info("Starting Metrics Viewer...");
		SpringApplication.run(MetricsViewerApplication.class, args);
		log.info("Metrics Viewer started successfully (port: 8080 by default)");
	}
}
