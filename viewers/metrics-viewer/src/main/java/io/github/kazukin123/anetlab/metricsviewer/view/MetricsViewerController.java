package io.github.kazukin123.anetlab.metricsviewer.view;

import java.util.List;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

import io.github.kazukin123.anetlab.metricsviewer.service.MetricsService;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetRunsResponse;

@RestController
@RequestMapping("/api")
public class MetricsViewerController {

    private final MetricsService metricsService;

    public MetricsViewerController(MetricsService metricsService) {
        this.metricsService = metricsService;
    }

    /** Run一覧 */
    @GetMapping("/runs.json")
    public GetRunsResponse getRuns() {
        return metricsService.getRuns();
    }

    /** 全データ取得 */
    @PostMapping(value="/metrics.json", consumes="application/x-www-form-urlencoded")
    @ResponseBody
    public GetMetricsResponse getMetrics(
            @RequestParam(required=false) List<String> runIds,
            @RequestParam(required=false) List<String> tagKeys) {
        GetMetricsRequest req = new GetMetricsRequest();
        if (runIds != null) req.getRunIds().addAll(runIds);
        if (tagKeys != null) req.getTagKeys().addAll(tagKeys);
        return metricsService.getMetrics(req);
    }
}
