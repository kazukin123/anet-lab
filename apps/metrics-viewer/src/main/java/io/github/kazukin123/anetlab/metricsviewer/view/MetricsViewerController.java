package io.github.kazukin123.anetlab.metricsviewer.view;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.http.ResponseEntity;

import io.github.kazukin123.anetlab.metricsviewer.service.MetricsApiException;
import io.github.kazukin123.anetlab.metricsviewer.service.MetricsService;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetRunsResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.PrioritizeRunsRequest;

@RestController
@RequestMapping("/api")
public class MetricsViewerController {

	private static final Logger log = LoggerFactory.getLogger(MetricsViewerController.class);

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
	public ResponseEntity<?> getMetrics(
			@RequestBody GetMetricsRequest request,
			@RequestHeader(value = "X-Query-Channel", required = false) String queryChannel,
			@RequestHeader(value = "X-Query-Sequence", required = false) String querySequence) {
		try {
			final GetMetricsResponse response = metricsService.getMetrics(
					request, queryChannel, querySequence);
			log.debug("getMetrics() series={}",
					request.getSeries() == null ? 0 : request.getSeries().size());
			return ResponseEntity.ok(response);
		} catch (MetricsApiException e) {
			return ResponseEntity.status(e.getStatus())
					.headers(e.getHeaders())
					.body(e.getBody());
		}
	}

	/** 取り込み優先Run集合の全置換 */
	@PostMapping(value = "/runs/prioritize", consumes = "application/json")
	public ResponseEntity<?> prioritizeRuns(@RequestBody PrioritizeRunsRequest request) {
		try {
			metricsService.prioritizeRuns(request);
			return ResponseEntity.noContent().build();
		} catch (MetricsApiException e) {
			return ResponseEntity.status(e.getStatus())
					.headers(e.getHeaders())
					.body(e.getBody());
		}
	}
}
