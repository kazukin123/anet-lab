package io.github.kazukin123.anetlab.metricsviewer.controller;

import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import io.github.kazukin123.anetlab.metricsviewer.model.MetricSeries;
import io.github.kazukin123.anetlab.metricsviewer.model.RunInfo;
import io.github.kazukin123.anetlab.metricsviewer.service.MetricsService;

/**
 * REST controller for runs and metrics APIs.
 */
@RestController
@RequestMapping("/api")
public class MetricsViewerController {

    private final MetricsService metricsService;

    public MetricsViewerController(MetricsService metricsService) {
        this.metricsService = metricsService;
    }

    @GetMapping(value = "/runs", produces = MediaType.APPLICATION_JSON_VALUE)
    public List<RunInfo> listRuns() {
        return metricsService.listRuns();
    }

    @GetMapping(value = "/metrics/{runId}", produces = MediaType.APPLICATION_JSON_VALUE)
    public Map<String, Object> getMetrics(
            @PathVariable("runId") String runId,
            @RequestParam(name = "tag", required = false) String tag) {

        List<MetricSeries> series = metricsService.fetchSeries(runId, Optional.ofNullable(tag));
        return Map.of("runId", runId, "series", series);
    }
}
