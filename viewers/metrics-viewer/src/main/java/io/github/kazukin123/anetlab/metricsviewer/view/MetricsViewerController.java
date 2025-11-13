package io.github.kazukin123.anetlab.metricsviewer.view;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
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
		final GetRunsResponse resp = metricsService.getRuns();
		log.debug("getRuns() resp={}", resp);
		return resp;
	}

	/** Metricsデータ取得 */
	@PostMapping(value = "/metrics.json", consumes = "application/json")
	@ResponseBody
	public GetMetricsResponse getMetrics(@RequestBody GetMetricsRequest request) {
	    log.info("getMetrics() runTagMap.size={}", 
	            request.getRunTagMap() != null ? request.getRunTagMap().size() : 0);

	    final GetMetricsResponse response = metricsService.getMetrics(request);
		if (log.isTraceEnabled()) {
			log.trace("getMetrics() req={} resp={}", request, response);
		} else {
			log.debug("getMetrics() req={}", request);
		}

		return response;
	}
}
