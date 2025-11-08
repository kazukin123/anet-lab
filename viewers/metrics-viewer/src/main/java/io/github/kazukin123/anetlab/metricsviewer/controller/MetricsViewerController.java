package io.github.kazukin123.anetlab.metricsviewer.controller;

import java.util.List;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import io.github.kazukin123.anetlab.metricsviewer.dto.GetMetricsRequest;
import io.github.kazukin123.anetlab.metricsviewer.dto.GetMetricsResponse;
import io.github.kazukin123.anetlab.metricsviewer.dto.GetRunsResponse;
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
    public GetRunsResponse getRuns() {
        return metricsService.getRuns();
    }

    /** 全データ取得 */
    @GetMapping("/metrics}")
    public GetMetricsResponse getMetrics(
            @RequestParam("runIds") List<String> runIds,
            @RequestParam("tagKeys") List<String> tagKeys) {
    	GetMetricsRequest req = new GetMetricsRequest();
        req.setRunIds(runIds);
        req.setTagKeys(tagKeys);
        return metricsService.getMetrics(req);
    }
}
