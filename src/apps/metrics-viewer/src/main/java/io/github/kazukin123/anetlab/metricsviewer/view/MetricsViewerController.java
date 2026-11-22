package io.github.kazukin123.anetlab.metricsviewer.view;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;

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
import io.github.kazukin123.anetlab.metricsviewer.service.RunConfigService;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetRunsResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.PrioritizeRunsRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RangeStatsRequest;

@RestController
@RequestMapping("/api")
public class MetricsViewerController {

	private static final Logger log = LoggerFactory.getLogger(MetricsViewerController.class);

	private final MetricsService metricsService;
	private final RunConfigService runConfigService;
	private final ObjectReader requestReader;

	public MetricsViewerController(MetricsService metricsService, RunConfigService runConfigService,
			ObjectMapper mapper) {
		this.metricsService = metricsService;
		this.runConfigService = runConfigService;
		this.requestReader = mapper.readerFor(JsonNode.class)
				.with(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);
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

	/** Run ごとの設定を、要求された列順に返す。 */
	@PostMapping(value = "/config.json", consumes = "application/json")
	public ResponseEntity<?> getConfig(@RequestBody(required = false) String body) {
		final JsonNode request = parseRequest(body);
		if (request == null || !request.isObject() || request.size() != 1
				|| !request.path("runIds").isArray()) {
			return invalidRequest("Request must contain only a runIds array");
		}
		// 文字列以外の JSON 値を Run ID に変換しない。
		final List<String> runIds = new ArrayList<>();
		for (JsonNode id : request.path("runIds")) {
			if (!id.isTextual() || id.textValue().isBlank()) {
				return invalidRequest("runIds must contain only non-empty strings");
			}
			runIds.add(id.textValue());
		}
		return ResponseEntity.ok(runConfigService.getConfig(runIds));
	}

	/** 指定された閉区間の範囲統計を返す。 */
	@PostMapping(value = "/range-stats.json", consumes = "application/json")
	public ResponseEntity<?> getRangeStats(
			@RequestBody(required = false) String body,
			@RequestHeader(value = "X-Query-Channel", required = false) String queryChannel,
			@RequestHeader(value = "X-Query-Sequence", required = false) String querySequence) {
		final JsonNode request = parseRequest(body);
		if (request == null || !request.isObject() || request.size() != 1
				|| !request.path("series").isArray()) {
			return invalidRequest("Request must contain only a series array");
		}
		// 閉じた本文を検証し、値を暗黙に文字列や整数へ変換しない。
		final List<RangeStatsRequest> series = new ArrayList<>();
		for (JsonNode item : request.path("series")) {
			final JsonNode runId = item.path("runId");
			final JsonNode tagKey = item.path("tagKey");
			final JsonNode from = item.path("fromStep");
			final JsonNode to = item.path("toStep");
			if (!item.isObject() || item.size() != 4
					|| !runId.isTextual() || runId.textValue().isBlank()
					|| !tagKey.isTextual() || tagKey.textValue().isBlank()
					|| !from.isIntegralNumber() || !from.canConvertToLong()
					|| !to.isIntegralNumber() || !to.canConvertToLong()) {
				return invalidRequest("Invalid range stats series at index " + series.size());
			}
			series.add(new RangeStatsRequest(runId.textValue(), tagKey.textValue(),
					from.longValue(), to.longValue()));
		}
		try {
			return ResponseEntity.ok(Map.of("data", metricsService.getRangeStats(series, queryChannel, querySequence)));
		} catch (MetricsApiException e) {
			return ResponseEntity.status(e.getStatus()).headers(e.getHeaders()).body(e.getBody());
		}
	}

	private JsonNode parseRequest(String body) {
		if (body == null) return null;
		try {
			return requestReader.readValue(body);
		} catch (JsonProcessingException e) {
			return null;
		}
	}

	private ResponseEntity<?> invalidRequest(String message) {
		return ResponseEntity.badRequest().body(Map.of("code", "invalid_request", "message", message));
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
