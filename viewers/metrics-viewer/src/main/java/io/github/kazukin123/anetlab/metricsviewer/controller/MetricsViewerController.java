package io.github.kazukin123.anetlab.metricsviewer.controller;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import io.github.kazukin123.anetlab.metricsviewer.model.MetricSeries;
import io.github.kazukin123.anetlab.metricsviewer.model.RunInfo;
import io.github.kazukin123.anetlab.metricsviewer.service.MetricsService;

@RestController
@RequestMapping("/api")
public class MetricsViewerController {

    private final MetricsService metricsService;

    public MetricsViewerController(MetricsService metricsService) {
        this.metricsService = metricsService;
    }

    /** Run一覧 */
    @GetMapping("/runs")
    public List<RunInfo> listRuns() {
        return metricsService.listRuns();
    }

    /** 全データ取得 */
    @GetMapping("/metrics/{runId}")
    public List<MetricSeries> getMetrics(@PathVariable("runId") String runId) {
        return metricsService.fetchSeries(runId, Optional.empty());
    }

    /** タグ一覧 */
    @GetMapping("/metrics/{runId}/tags")
    public List<Map<String, String>> getTags(@PathVariable("runId") String runId) {
        List<MetricSeries> all = metricsService.fetchSeries(runId, Optional.empty());
        List<Map<String, String>> tags = new ArrayList<>();
        for (MetricSeries s : all) {
            tags.add(Map.of("tag", s.getTag()));
        }
        return tags;
    }
}
