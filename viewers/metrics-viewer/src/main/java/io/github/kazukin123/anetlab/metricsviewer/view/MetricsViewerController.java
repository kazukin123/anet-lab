package io.github.kazukin123.anetlab.metricsviewer.view;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

import io.github.kazukin123.anetlab.metricsviewer.service.MetricsRepository;
import io.github.kazukin123.anetlab.metricsviewer.service.MetricsService;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetRunsResponse;

@RestController
@RequestMapping("/api")
public class MetricsViewerController {

    private static final Logger log = LoggerFactory.getLogger(MetricsRepository.class);

    private final MetricsService metricsService;

    public MetricsViewerController(MetricsService metricsService) {
        this.metricsService = metricsService;
    }

    /** Run一覧 */
    @GetMapping("/runs.json")
    public GetRunsResponse getRuns() {
    	GetRunsResponse resp = metricsService.getRuns();
    	log.debug("getRuns() resp={}", resp);
    	return resp;
    }

    /** 全データ取得 */
    @PostMapping(value = "/metrics.json", consumes = "application/x-www-form-urlencoded")
    @ResponseBody
    public GetMetricsResponse getMetrics(
            @RequestParam(required=false) List<String> runIds,
            @RequestParam(required=false) List<String> tagKeys) {
        GetMetricsRequest req = new GetMetricsRequest();
        if (runIds != null) req.getRunIds().addAll(runIds);
        if (tagKeys != null) req.getTagKeys().addAll(tagKeys);
        GetMetricsResponse resp = metricsService.getMetrics(req);
        
        if (log.isTraceEnabled()) {
        	log.trace("getMetrics() req={} resp={}", req, resp);
        } else {
        	log.info("getMetrics() req={}", req);
        }
        return resp;
    }
}
